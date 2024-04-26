from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
from langchain_community.llms import Ollama
import os

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LLMInput(BaseModel):
    message: str

class FinancialAdvisor:
    def __init__(self):
        self.QDRANT_URL = os.environ["QDRANT_URL"]
        self.QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
        self.OPENAI_API_BASE = os.environ["OPENAI_API_BASE"]
        self.OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
        self.COLLECTION_NAME = "financial-docs"
        self.index, self.retriever = self.get_index_and_retriever()
        self.llm = FireworkLLM()
        self.client = plaid.ApiClient(plaid.Configuration(
            host=plaid.Environment.Sandbox,  # Use the appropriate environment
            api_key={
                'clientId': os.environ["PLAID_CLIENT_ID"],
                'secret': os.environ["PLAID_SECRET"],
            }
        ))
        self.plaid_client = plaid_api.PlaidApi(self.client)
        #self.access_token = os.environ["PLAID_ACCESS_TOKEN"]
        self.access_token = get_access_token()
        self.agent_executor = self.get_agent_executor()

    def get_dummy_transactions(self):
        return {
            "transactions": [
                {
                    "account_id": "account1",
                    "amount": 150.00,
                    "category": ["Food and Drink", "Restaurants"],
                    "date": "2023-09-28",
                    "name": "Starbucks"
                },
                {
                    "account_id": "account1",
                    "amount": 200.00,
                    "category": ["Travel", "Airlines"],
                    "date": "2023-09-20",
                    "name": "Delta Airlines"
                },
                {
                    "account_id": "account2",
                    "amount": 1200.00,
                    "category": ["Payment", "Rent"],
                    "date": "2023-09-01",
                    "name": "Monthly Rent"
                },
                {
                    "account_id": "account2",
                    "amount": 300.00,
                    "category": ["Shops", "Electronics"],
                    "date": "2023-09-15",
                    "name": "Best Buy"
                }
            ]
        }

    def get_access_token():
        # Create a sandbox public token for a test institution
        institution_id = 'ins_1'  # Use a Sandbox institution ID
        initial_products = ['transactions']
        options = {}  # Add any required options here

        # Create the request data as a dictionary
        request_data = {
            'institution_id': institution_id,
            'initial_products': initial_products,
            'options': options
        }

        try:
            # Pass the request data as a single dictionary argument
            pt_response = plaid_client.sandbox_public_token_create(request_data)
            public_token = pt_response['public_token']

            # Exchange the public token for an access token
            exchange_response = plaid_client.item_public_token_exchange({'public_token': public_token})
            access_token = exchange_response['access_token']
            return access_token
        except plaid.ApiException as e:
            print("An error occurred while exchanging public token:", e)
            return None
    
    def get_transactions(self, start_date: str, end_date: str) -> Dict:
          """Fetch transactions from Plaid or use dummy data for testing."""
          try:
              # response = self.plaid_client.transactions_get({
              #     'access_token': self.access_token,
              #     'start_date': start_date,
              #     'end_date': end_date,
              #     'options': {
              #         'count': 250,
              #         'offset': 0
              #     }
              # })
              # return response.to_dict()
              return self.get_dummy_transactions()
          except plaid.ApiException as e:
              return e.body

    def analyze_transactions(self, transactions):
      # Grouping expenses by category
      category_expenses = {}
      for transaction in transactions['transactions']:
          for category in transaction['category']:
              if category not in category_expenses:
                  category_expenses[category] = 0
              category_expenses[category] += transaction['amount']

      advice = []
      # Setting budget thresholds for categories
      budget_thresholds = {
          "Restaurants": 100,
          "Airlines": 300,
          "Rent": 1200,  # Example fixed monthly rent
          "Electronics": 200
      }

      # Generating advice based on spending compared to budget
      for category, spent in category_expenses.items():
          budget = budget_thresholds.get(category, None)
          if budget:
              if spent > budget:
                  advice.append(f"Consider reducing your spending on {category}. You spent ${spent}, which is over your budget of ${budget}.")
              else:
                  advice.append(f"Your spending on {category} is within budget at ${spent}. Good job managing your expenses!")
          else:
              advice.append(f"Your spending on {category} was ${spent}. Consider setting a budget for this category.")

      return advice

    def summarize_transactions(self, transactions):
        """Prepare a summary of transactions to feed into the AI for generating advice."""
        category_expenses = {}
        for transaction in transactions['transactions']:
            for category in transaction['category']:
                if category not in category_expenses:
                    category_expenses[category] = 0
                category_expenses[category] += transaction['amount']
        return category_expenses

    def provide_financial_advice(self, start_date, end_date):
        transactions = self.get_transactions(start_date, end_date)
        if 'transactions' in transactions:
            category_expenses = self.summarize_transactions(transactions)
            query = ", ".join(f"{category}: ${spent}" for category, spent in category_expenses.items())
            advice = self.agent_executor.run(query)  # Utilize the agent to analyze and provide advice
            return advice
        else:
            return ["Error retrieving transactions"]

    def get_index_and_retriever(self):
        qdrant_client = QdrantClient(url=self.QDRANT_URL, api_key=self.QDRANT_API_KEY)
        embed_model = OpenAIEmbedding(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            api_base=os.environ["OPENAI_API_BASE"],
            api_key=os.environ["OPENAI_API_KEY"])
        Settings.embed_model = embed_model

        vector_store = QdrantVectorStore(client=qdrant_client, collection_name=self.COLLECTION_NAME)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        retriever = index.as_retriever()
        return index, retriever

    def get_relevant_documents(self, query: str) -> List[str]:
        documents = self.retriever.retrieve(query)
        sorted_documents = sorted(documents, key=lambda node: node.score, reverse=True)
        relevant_documents = [node.text.replace("\n", " ") for node in sorted_documents]
        return relevant_documents

    def provide_financial_analysis(self, question: str) -> List[str]:
        research_template = PromptTemplate.from_template(
            template="""
        You are a financial analyst providing advice to clients. You have expertise in financial markets and personal finance.
        A client is asking the following question:
        {question}

        Your task is to provide a list of the top 5 financial insights or advice points. The list should be prioritized based on factors that can predict financial outcomes and should be simple, concise.
        This list will be used to guide the client's financial decisions.

        Respond in a `-` delimited list format with in-depth, independently operable advice points.
        """
        )

        prompt = research_template.format(
            question=question,
        )

        response = self.llm.invoke(prompt)
        advice = list(
                filter(
                    lambda y: y != "",
                    map(
                        lambda x: x.strip()\
                                    .replace("- ", "")\
                                    .replace("\"", ""),
                        response.split("\n")
                        )
                    )
                )
        return advice

    def get_agent_executor(self) -> AgentExecutor:
        retriever_tool = create_retriever_tool(
            self.retriever,
            "document_retriever",
            "Search for information on the latest financial news and market research. For any questions about market trends or financial advice, use this tool.",
        )

        tools = [
            retriever_tool,
        ]

        agent_prompt = PromptTemplate(
            input_variables=['agent_scratchpad', 'input', 'tool_names', 'tools'],
            template="""
            Answer the following questions as best you can.
            You have access to the following tools that can reference the latest research on the given topic.
            Do not hallucinate.  If you do not have the relevant research to the question, exclusively say `NOT ENOUGH INFORMATION`:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}"""
        )

        agent = create_openai_tools_agent(
            self.llm, tools, agent_prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            handle_parsing_errors=True,
            max_execution_time=600
        )

        return agent_executor

    def run(self):
      today = datetime.today()
      start_date = (today - timedelta(days=30)).strftime('%Y-%m-%d')
      end_date = today.strftime('%Y-%m-%d')

      transactions = self.get_transactions(start_date, end_date)
      advice = self.analyze_transactions(transactions)

      print("\n".join(advice))
    

@app.get("/")
def read_root():
    return {"message": "Connected to FastAPI"}

@app.post("/download_model/{model}")
async def download_model(model):
    body = {
        "name": model,
    }
    response = requests.post(os.environ["OLLAMA_API_URL"] + "/api/pull", json=body)
    if response.ok:
        return "Pulled " + model
    else:
        return "Failed to pull model."

@app.post("/generate/")
async def llm_generate(input: LLMInput):
    llm = Ollama(model="mistral", base_url=os.environ["OLLAMA_API_URL"])

    return llm.invoke(input.message)

@app.post("/financial_advice/")
async def get_financial_advice(start_date: str, end_date: str):
    advisor = FinancialAdvisor()
    advice = advisor.provide_financial_advice(start_date, end_date)
    return {"advice": advice}