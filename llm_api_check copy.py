from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result = generator("Hello, I'm a language model,", max_length=30)
print (result[0]['generated_text'])

# def generate_response(query, context):
#     prompt = f"Ebook content:\n{context}\n\nUser query: {query}\n\nAnswer:"
#     response = openai.Completion.create(
#         engine="text-davinci-003",
#         prompt=prompt,
#         max_tokens=200
#     )
#     return response.choices[0].text.strip()