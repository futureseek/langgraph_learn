import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from langchain.schema import Document
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    CharacterTextSplitter
)
import pypdf
import docx

# Meta-Chunking imports
import torch
import torch.nn.functional as F
from nltk.tokenize import sent_tokenize
try:
    import jieba
except ImportError:
    jieba = None


class MetaChunking:
    """Meta-Chunking实现类，包含多种分块策略"""
    
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
    
    def get_ppl_batch(self, input_ids=None, attention_mask=None, past_key_values=None, return_kv=False, end=None):
        """计算批量文本的困惑度"""
        past_length = 0
        if end is None:
            end = input_ids.shape[1]
        with torch.no_grad():
            response = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = response.past_key_values
        shift_logits = response.logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., past_length + 1: end].contiguous()
        # Flatten the tokens
        active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
        active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
        active_labels = shift_labels.view(-1)[active]
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(active_logits, active_labels)
        res = loss
        return (res, past_key_values) if return_kv else res
    
    def split_text_by_punctuation(self, text, language):
        """根据标点符号分割文本"""
        if language == 'zh':
            if jieba is None:
                # 如果没有jieba，使用简单的分割方式
                sentences = re.split(r'[。！？；]', text)
                sentences = [s.strip() for s in sentences if s.strip()]
                return sentences
            else:
                sentences = jieba.cut(text, cut_all=False)
                sentences_list = list(sentences)
                sentences = []
                temp_sentence = ""
                for word in sentences_list:
                    if word in ["。", "！", "？", "；"]:
                        sentences.append(temp_sentence.strip() + word)
                        temp_sentence = ""
                    else:
                        temp_sentence += word
                if temp_sentence:
                    sentences.append(temp_sentence.strip())
                return sentences
        else:
            full_segments = sent_tokenize(text)
            ret = []
            for item in full_segments:
                item_l = item.strip().split(' ')
                if len(item_l) > 512:
                    if len(item_l) > 1024:
                        item = ' '.join(item_l[:256]) + "..."
                    else:
                        item = ' '.join(item_l[:512]) + "..."
                ret.append(item)
            return ret
    
    def find_minima(self, values, threshold):
        """找到局部最小值点"""
        minima_indices = []
        for i in range(1, len(values) - 1):
            if values[i] < values[i - 1] and values[i] < values[i + 1]:
                if (values[i - 1] - values[i] >= threshold) or (values[i + 1] - values[i] >= threshold):
                    minima_indices.append(i)
            elif values[i] < values[i - 1] and values[i] == values[i + 1]:
                if values[i - 1] - values[i] >= threshold:
                    minima_indices.append(i)
        return minima_indices
    
    def perplexity_chunking(self, text, threshold=1.0, language='en') -> List[str]:
        """基于困惑度的分块策略"""
        if self.model is None or self.tokenizer is None:
            # 如果没有模型，回退到递归字符分割
            return self._fallback_chunking(text)
        
        try:
            # 分割文本为句子
            segments = self.split_text_by_punctuation(text, language)
            segments = [item for item in segments if item.strip()]
            
            if len(segments) <= 1:
                return [text]
            
            # 计算每个句子的困惑度
            len_sentences = []
            input_ids = torch.tensor([[]], device=self.model.device, dtype=torch.long)
            attention_mask = torch.tensor([[]], device=self.model.device, dtype=torch.long)
            
            for context in segments:
                tokenized_text = self.tokenizer(context, return_tensors="pt", add_special_tokens=False)
                input_id = tokenized_text["input_ids"].to(self.model.device)
                input_ids = torch.cat([input_ids, input_id], dim=-1)
                len_sentences.append(input_id.shape[1])
                attention_mask_tmp = tokenized_text["attention_mask"].to(self.model.device)
                attention_mask = torch.cat([attention_mask, attention_mask_tmp], dim=-1)
            
            loss, past_key_values = self.get_ppl_batch(
                input_ids,
                attention_mask,
                past_key_values=None,
                return_kv=True
            )
            
            first_cluster_ppl = []
            index = 0
            for i in range(len(len_sentences)):
                if i == 0:
                    first_cluster_ppl.append(loss[0:len_sentences[i] - 1].mean().item())
                    index += len_sentences[i] - 1
                else:
                    first_cluster_ppl.append(loss[index:index + len_sentences[i]].mean().item())
                    index += len_sentences[i]
            
            # 找到局部最小值点作为分块边界
            minima_indices = self.find_minima(first_cluster_ppl, threshold)
            
            # 根据分块边界分割文本
            split_points = [0] + minima_indices + [len(first_cluster_ppl) - 1]
            final_chunks = []
            
            for i in range(len(split_points) - 1):
                tmp_sentence = []
                if i == 0:
                    tmp_sentence.append(segments[0])
                for sp_index in range(split_points[i] + 1, split_points[i + 1] + 1):
                    tmp_sentence.append(segments[sp_index])
                final_chunks.append(''.join(tmp_sentence))
            
            return final_chunks
        except Exception as e:
            print(f"困惑度分块失败，回退到默认分块: {e}")
            return self._fallback_chunking(text)
    
    def prob_subtract_chunking(self, text, threshold=0, language='en') -> List[str]:
        """基于概率差的分块策略"""
        if self.model is None or self.tokenizer is None:
            # 如果没有模型，回退到递归字符分割
            return self._fallback_chunking(text)
        
        try:
            full_segments = self.split_text_by_punctuation(text, language)
            full_segments = [item for item in full_segments if item.strip()]
            
            if len(full_segments) <= 1:
                return [text]
            
            save_list = []
            tmp = ''
            threshold_list = []
            
            for i, sentence in enumerate(full_segments):
                if tmp == '':
                    tmp += sentence
                else:
                    prob_subtract = self.get_prob_subtract(tmp, sentence, language)
                    threshold_list.append(prob_subtract)
                    
                    if prob_subtract > threshold:
                        if language == 'zh':
                            tmp += sentence
                        else:
                            tmp += ' ' + sentence
                    else:
                        save_list.append(tmp)
                        tmp = sentence
                
                if len(threshold_list) >= 5:
                    last_five = threshold_list[-5:]
                    avg = sum(last_five) / len(last_five)
                    threshold = avg
            
            if tmp != '':
                save_list.append(tmp)
            
            return save_list if save_list else [text]
        except Exception as e:
            print(f"概率差分块失败，回退到默认分块: {e}")
            return self._fallback_chunking(text)
    
    def get_prob_subtract(self, sentence1, sentence2, language):
        """计算两个句子的概率差"""
        try:
            if language == 'zh':
                query = '''这是一个文本分块任务.你是一位文本分析专家，请根据提供的句子的逻辑结构和语义内容，从下面两种方案中选择一种分块方式：
                1. 将"{}"分割成"{}"与"{}"两部分；
                2. 将"{}"不进行分割，保持原形式；
                请回答1或2。'''.format(sentence1 + sentence2, sentence1, sentence2, sentence1 + sentence2)
                prompt = "