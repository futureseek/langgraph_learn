from openai import OpenAI

client = OpenAI(
    base_url='https://ms-ens-13e385aa-7c44.api-inference.modelscope.cn/v1',
    api_key='ms-8b59067c-75ff-4b83-900e-26e00e46c531', # ModelScope Token
)

response = client.embeddings.create(
    model='Qwen/Qwen3-Embedding-0.6B-GGUF', # ModelScope Model-Id, required
    input='你好',
    encoding_format="float"
)

print(response)