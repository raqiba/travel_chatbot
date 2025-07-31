# # main.py

# import os
# import json
# import re
# import datetime
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain.chains import RetrievalQA
# from langchain.vectorstores import FAISS
# from langchain.document_loaders import TextLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain.agents import initialize_agent, Tool, AgentType
# from langchain_core.prompts.prompt import PromptTemplate

# # Load .env file
# load_dotenv()

# def load_flights():
#     with open('data/flights.json') as f:
#         return json.load(f)



# def setup_retrieval_qa():
#     load_dotenv()

#     docs = TextLoader("data/visa_rules.md").load()
#     chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

#     embeddings = GoogleGenerativeAIEmbeddings(
#         model="models/gemini-embedding-001",
#         task_type="RETRIEVAL_DOCUMENT",
#         google_api_key=os.getenv("OPENAI_API_KEY")
#     )
#     vectorstore = FAISS.from_documents(chunks, embeddings)
#     retriever = vectorstore.as_retriever()

#     # Single correct instantiation:
#     prompt = PromptTemplate(
#         template=(
#             "You are a travel policy assistant. You need to assist people by answering them questions related to their travel.\n"
#             "For the questions asked which are not this context, you need to say:"
#             "I am a Travel policy Assistant, please ask the question this domain."
#             "Some examples of the questions are: Find me a round-trip to Tokyo in august with Star Alliance airlines only. I want to avoid overnight layovers."
#             "Use strictly this context:\n{context}\n"
#             "User question: {query}\n"
#             "If you don’t know, say \"I don't know\"."
#         ),
#         input_variables=["context", "query"]
#     )

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0,
#         google_api_key=os.getenv("OPENAI_API_KEY")
#     )

#     return RetrievalQA.from_chain_type(
#         llm=llm,
#         retriever=retriever,
#         chain_type_kwargs={"prompt": prompt}
#     )

# def create_flight_search_tool(flights_data):
#     def search_flights(parsed):
#         origin = parsed.get("from")
#         destination = parsed.get("to")
#         month = parsed.get("departure_month")
#         prefers_star = parsed.get("airline_alliance", "").lower() == "star alliance"
#         avoid_overnight = parsed.get("avoid_overnight", False)

#         def is_month_match(date_str, target_month):
#             try:
#                 date = datetime.strptime(date_str, "%Y-%m-%d")
#                 return date.strftime("%B") == target_month
#             except:
#                 return False

#         results = []
#         for flight in flights_data:
#             if origin and flight["from"].lower() != origin.lower():
#                 continue
#             if destination and flight["to"].lower() != destination.lower():
#                 continue
#             if prefers_star and flight["alliance"].lower() != "star alliance":
#                 continue
#             if avoid_overnight and any("overnight" in lay.lower() for lay in flight["layovers"]):
#                 continue
#             if month and not is_month_match(flight["departure_date"], month):
#                 continue
#             results.append(flight)

#         if not results:
#             return "❌ No flights match your criteria."

#         output = ""
#         for i, f in enumerate(results, start=1):
#             output += (
#                 f"**Flight {i}**  \n"
#                 f"✈️ Airline: {f['airline']}  \n"
#                 f"Route: {f['from']} → {f['to']}  \n"
#                 f"Dates: {f['departure_date']} → {f['return_date']}  \n"
#                 f"Price: ${f['price_usd']}  \n"
#                 f"Refundable: {'Yes' if f['refundable'] else 'No'}  \n"
#                 f"Layovers: {', '.join(f['layovers']) or 'Direct'}  \n"
#                 f"---\n"
#             )
#         return output
#     return search_flights


# def create_agent():
#     flights = load_flights()
#     qa = setup_retrieval_qa()
#     tools = [
#         Tool(name="FlightSearch", func=create_flight_search_tool(flights), description="Search flights"),
#         Tool(name="VisaQA", func=qa.run, description="Visa/refund QA")
#     ]
#     llm = ChatGoogleGenerativeAI(
#         model="gemini-2.5-flash",
#         temperature=0,
#         google_api_key=os.getenv("OPENAI_API_KEY")
#     )
#     agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
#     return agent

# if __name__ == "__main__":
#     agent = create_agent()
#     print("Chatbot ready (Gemini)!")
#     while True:
#         user = input("You: ")
#         if user.lower() in ("exit", "quit"):
#             break
#         print("Bot:", agent.run(user))


# main.py

import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Load flight data from JSON
def load_flights():
    with open("data/flights.json", "r") as f:
        return json.load(f)

# Set up Retrieval QA for visa/refund queries
def setup_retrieval_qa():
    docs = TextLoader("data/visa_rules.md").load()
    chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=0).split_documents(docs)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="RETRIEVAL_DOCUMENT",
        google_api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = PromptTemplate(
        template="""
        You are a travel policy assistant. Answer only travel-related questions (visa rules, refund policy, etc).
        If the question is unrelated, say: "I am a travel assistant and only answer travel-related questions."
        Use this context:\n{context}\n
        User question: {query}\n
        If unsure, say "I don't know."
        """,
        input_variables=["context", "query"]
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("OPENAI_API_KEY")
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

# Create LLM chain to parse user's flight query into structured JSON
def create_flight_parser_chain(llm):
    prompt = PromptTemplate.from_template("""
        You are a helpful assistant that extracts flight search criteria from natural language.
        populate the Json given in example for the available entities, the ones which are not given should be given a value null.
        the departure and destination can be cities, countries or states etc. 
        Only output a **valid JSON object** with the following keys:
        - "from": departure city or null
        - "to": destination city or null
        - "departure_month": e.g., "August" or null
        - "airline_alliance": e.g., "Star Alliance" or null
        - "avoid_overnight": true or false

        Even if the input is vague, return best-guess JSON with null where unknown.
        Do NOT ask clarifying questions. Do NOT include explanations.

        Example 1:
        Input: Find me a round-trip to Tokyo in August with Star Alliance airlines only. I want to avoid overnight layovers.
        Output:
        {{
        "from": null,
        "to": "Tokyo",
        "departure_month": "August",
        "airline_alliance": "Star Alliance",
        "avoid_overnight": true
        }}
        
        Example 2:
        Input: Find me a flight from new york to india.
        Output:
        {{
        "from": new york,
        "to": "india",
        "departure_month": "null",
        "airline_alliance": "null",
        "avoid_overnight": null
        }}

        Now do this:
        Input: {query}
        Output:
            """)
    return LLMChain(llm=llm, prompt=prompt)

# Search flights based on structured criteria
def create_flight_search_tool(flights_data):
    def search_flights(parsed):
        if isinstance(parsed, str):
            try:
                parsed = json.loads(parsed)
            except:
                return "❌ Error: Couldn't parse the search criteria."

        origin = parsed.get("from")
        destination = parsed.get("to")
        month = parsed.get("departure_month")
        alliance = parsed.get("airline_alliance", "").lower()
        avoid_overnight = parsed.get("avoid_overnight", False)

        def is_month_match(date_str, target_month):
            try:
                date = datetime.strptime(date_str, "%Y-%m-%d")
                return date.strftime("%B").lower() == target_month.lower()
            except:
                return False

        results = []
        for flight in flights_data:
            if origin and flight["from"].lower() != origin.lower():
                continue
            if destination and flight["to"].lower() != destination.lower():
                continue
            if alliance and flight["alliance"].lower() != alliance:
                continue
            if avoid_overnight and any("overnight" in lay.lower() for lay in flight["layovers"]):
                continue
            if month and not is_month_match(flight["departure_date"], month):
                continue
            results.append(flight)

        if not results:
            return "❌ No flights match your criteria."

        output = ""
        for i, f in enumerate(results, start=1):
            output += (
                f"**Flight {i}**  \n"
                f"✈️ Airline: {f['airline']}  \n"
                f"Route: {f['from']} → {f['to']}  \n"
                f"Dates: {f['departure_date']} → {f['return_date']}  \n"
                f"Price: ${f['price_usd']}  \n"
                f"Refundable: {'Yes' if f['refundable'] else 'No'}  \n"
                f"Layovers: {', '.join(f['layovers']) or 'Direct'}  \n"
                f"---\n"
            )
        return output
    return search_flights

# Build the full agent with tools
def create_agent():
    flights = load_flights()

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("OPENAI_API_KEY")
    )

    qa = setup_retrieval_qa()
    parser_chain = create_flight_parser_chain(llm)
    flight_search = create_flight_search_tool(flights)

    def full_flight_handler(query):
        parsed = parser_chain.run({"query": query})
        return flight_search(parsed)

    tools = [
        Tool(name="FlightSearch", func=full_flight_handler, description="Search for flights based on natural query"),
        Tool(name="VisaQA", func=qa.run, description="Answer visa and refund policy questions")
    ]

    return initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True  # ✅ Add this
)

# CLI loop
if __name__ == "__main__":
    agent = create_agent()
    print("🌍 International Travel Assistant is ready!\n(Type 'exit' to quit)\n")
    while True:
        user = input("You: ")
        if user.lower() in ("exit", "quit"):
            break
        try:
            response = agent.run(user)
            print("✈️ Bot:", response)
        except Exception as e:
            print("⚠️ Error:", e)
