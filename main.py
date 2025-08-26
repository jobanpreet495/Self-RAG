from self_rag import SelfRAGModel

s = SelfRAGModel(urls = ["https://weaviate.io/blog/what-is-agentic-rag"])
response = s.generate("what are the core components of an AI agent mentioned in this blog?")
print(response)