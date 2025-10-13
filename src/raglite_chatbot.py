
    from sentence_transformers import SentenceTransformer
    import faiss, json, numpy as np
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Load model + tokenizer
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Load retrieval encoder
    retriever = SentenceTransformer('all-MiniLM-L6-v2')

    # Load FAQ knowledge base
    with open('data/faq_data.json') as f:
        kb = json.load(f)

    corpus = [item['question'] + ' ' + item['answer'] for item in kb]
    corpus_embeddings = retriever.encode(corpus)

    # Create FAISS index
    index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
    index.add(np.array(corpus_embeddings, dtype='float32'))

    def retrieve_context(query, k=2):
        q_emb = retriever.encode([query])
        D, I = index.search(np.array(q_emb, dtype='float32'), k)
        return [corpus[i] for i in I[0]]

    def rag_answer(query):
        contexts = retrieve_context(query)
        combined = query + "

Relevant Info:
" + "
".join(contexts)
        inputs = tokenizer(combined, return_tensors='pt')
        outputs = model.generate(**inputs, max_new_tokens=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    if __name__ == '__main__':
        q = input('Ask a travel question: ')
        print('Answer:', rag_answer(q))
