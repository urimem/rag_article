import pandas
from openai import OpenAI
import numpy
from log import log 

client = OpenAI()

# Generate text embedding 
def get_embedding(text):
  text = text.replace("\n", " ")
  return client.embeddings.create(input = [text], model = "text-embedding-3-small").data[0].embedding

# Create & add embeddings to films CSV
df = pandas.read_pickle("data/films-with-embeddings.pkl")
df.head()

# Conditional generation of embedding to each film synopsisi
if not 'embeddings' in df.columns:
  df['embeddings'] = df['synopsis'].apply(get_embedding)
  df.to_pickle('data/films-with-embeddings.pkl')  # pkl format is lighter than CSV

# Creating embeddings for user request/prompt
user_request = "I am looking for a film about LGBTQ"
user_request_embedding = get_embedding(user_request)

# Calculating similarity/distance from user_request to each film
def get_similarity(film_embedding):
  return numpy.dot(film_embedding, user_request_embedding)

# Adding similarity column 
df['similarity'] = df['embeddings'].apply(get_similarity)

# Sort the records by similarity and take the first X records
df.sort_values('similarity', ascending = False, inplace = True)

prompt_context = \
"1.\nFilm name: " + df['film_title'].iloc[0] + "\nSynopsis:\n" + df['synopsis'].iloc[0] + "\n" + \
"2.\nFilm name: " + df['film_title'].iloc[1] + "\nSynopsis:\n" + df['synopsis'].iloc[1] + "\n" + \
"3.\nFilm name: " + df['film_title'].iloc[2] + "\nSynopsis:\n" + df['synopsis'].iloc[2] + "\n"

prompt_messages = [
    {"role": "system", "content": "You are a film expert helping ""Film Platform"", a documentary film distribution company, respond to educators looking for the right film to use for teaching."},
    {"role": "user", "content": user_request},
    {"role": "assistant", "content": f"Use this information from Film Platform's database to answer the user question: {prompt_context}. Please stick to the context when answering the question and recommend a maximum of two films."}
  ]

rag_model = "gpt-4o"

# Model generation call
response = client.chat.completions.create(model = rag_model, messages = prompt_messages)
print (response.choices[0].message.content)

log("model:\n" + rag_model)
log("prompt_messages:\n" + prompt_messages[0]["content"] + "\n" + prompt_messages[1]["content"] + "\n" + prompt_messages[2]["content"] + "\n")
log("response:\n" + response.choices[0].message.content)
log("End\n\n")
