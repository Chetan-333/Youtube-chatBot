

from langchain_google_genai import ChatGoogleGenerativeAI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



from dotenv import load_dotenv
load_dotenv()
model =ChatGoogleGenerativeAI(model ="gemini-2.5-flash" )


from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate







video_id = "ckNEdxQ0Tc0"
 # only the ID, not full URL

try:
    # Fetch the transcript
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    
    # Convert to list if it's a generator/iterator
    if not isinstance(transcript_list, list):
        transcript_list = list(transcript_list)

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    #print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

#print(transcript_list)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = text_splitter.create_documents([transcript])
#print(len(chunks))




from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

#print(vectorstore.index_to_docstore_id)

#step-2 Retrieval

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

#print(retriever.invoke("What is the video about?"))


#Augmentation

prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

#print("Retrieved docs:", retrieved_docs)

context_text = "\n".join([doc.page_content for doc in retrieved_docs])  

final_prompt = prompt.format(context=context_text, question=question)     

#generate final answer

answer = model.invoke(final_prompt)
#print("Answer:", answer.content)

#Chain

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parser = StrOutputParser()
main_chain = parallel_chain | prompt | model| parser

response = main_chain.invoke(question)
print("Final Response:", response)

#chorme plugin