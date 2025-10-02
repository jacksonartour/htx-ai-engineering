# htx-ai-engineering
Take home test for ai engineering. This README documents the project requirements, dependencies and approach on the various tasks for the project. 

Refer to the ai-agents.ipynb file for the full execution documentation

## Project structure

```
htx-ai-engineering/
├── .env
├── .gitignore
├── AI Engineering Take Home Test v2.0.pdf
├── ai-agents.ipynb
├── fy2024_analysis_of_revenue_and_expenditure.pdf
├── README.md
├── requirements.txt
└── template.env
```

## Pre-requisites
- Python 3.1x
- Gemini API key <- Will be included in the email

Package depencies are listed in requirements.txt. 

## Task 1: Document extraction & prompt engineering 
My approach for this task was to use pymupdf to read text from the pdf document by page, and then prompt it using separate prompts. 

For this task I decided to use 4 prompts instead of 1 generic prompt, to precisely parse the data and customise the prompt to the required data. The approach was nothing fancy, just a system prompt to define the message, and a human message specific to the task for extraction.

```
# Example prompt used for corporate tax incomes 
corp_income_tax_prompt = """
Extract the exact amount of Corporate Income Tax specified from the following page text. 
Assume units is in billions, return only the number as a float.

PAGE TEXT:
{page_text}
"""
```

We then parse this as the system message and then parse the text (page specific) into the LLM

```
corp_income_tax_resp = llm.invoke([
    SystemMessage(content=DEFAULT_SYSTEM_MESSAGE),
    HumanMessage(content=corp_income_tax_prompt.format(page_text=pdf_text[5]))
]).content
print(f"Corporate Income Tax Response: {corp_income_tax_resp}")
try:
    corporate_income_tax_2024 = float(corp_income_tax_resp)
except Exception:
    corporate_income_tax_2024 = None
```

This approach precisely gathers the data from the text, and we can parse it directly into a structured format. 

```
# Example output from the completed task
Corporate Income Tax Response: 28.4
YOY Corporate Income Tax Response: 17.0
Total Top-ups Response: 20.352
Operating Revenue Taxes Response: Here's a list of the tax names mentioned in the "Operating Revenue" section of the provided text:

*   "Corporate Income Tax"
*   "Other Taxes"
*   "Personal Income Tax"
*   "Assets Taxes"
*   "Betting Taxes"
*   "Goods and Services Tax"
*   "Water Conservation Tax"
*   "Annual Tonnage Tax"
*   "Casino Taxes"
```

## Task 2: Tool calling & reasoning integration
For this task, I opted to use the @tool decorator in the interest of time, and my admittedly, limited knowledge on integrating local MCPs. 

The challenge for this task was to come up with a prompt such that the agent can exactly use the tool. 

In this task, the limitation was on the model used. As I opted for gemini-2.5-flash, it may not be the latest model and may not recognise tool usage properly. The prompt was then very customised to ensure that the tool was called and prevent the LLM from coming up with information on it's own. 

### How to ensure structured output? 
I used a pydantic output parser to convert the JSON output from the LLM into a usable class. 

```
output_parser = PydanticOutputParser(pydantic_object=DateExtractionOutput)

# Create the agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Wrapper function to parse agent output
def process_with_structured_output(text: str) -> DateExtractionOutput:
    result = agent_executor.invoke({"input": text})
    # Parse the agent's output into structured format
    return output_parser.parse(result["output"])

result_structured = process_with_structured_output("The submission was on 15 March 2024")
```

```
# Example output when invoking agent
> Entering new AgentExecutor chain...

Invoking: `normalize_submission_date` with `{'date_str': '15 March 2024'}`


2024-03-15
{
    "original_text": "The submission was on 15 March 2024",
    "normalized_date": "2024-03-15",
    "status": "Upcoming"
}


> Finished chain.
```

### Limitations
As the model used may not be the latest, it will struggle with a larger amount of text. Refer to the outputs of page 1 vs page 36. Page 36 failed to call the tool. However, I believe that using a more capable model may resolve this problem. 


## Task 3: Multi-Agent supervisor systems
For this task, I followed a 3 step approach.

1. Define a state to store information in the graph
2. Create agent nodes with specific responsibilities 
3. Define workflow graph to run a query. 

### State creation
To keep it simple, the state keeps information from various nodes in the graph refer below to see the variables used

```
class BudgetAnalysisState(TypedDict):
    """
    Shared state that flows through all agents in the graph.
    Each agent reads from and writes to this state.
    """
    
    # Original user query
    query: str
    
    # Supervisor's routing decision
    supervisor_decision: Annotated[list[str], operator.add]  # Can accumulate decisions
    
    # Revenue agent's findings
    revenue_info: str  
    
    # Expenditure agent's findings
    expenditure_info: str  
    
    # Final aggregated answer
    final_answer: str
    
    # Messages for conversation history (inherited from MessagesState pattern)
    messages: Annotated[list[BaseMessage], operator.add]
```


### Create agent nodes with specific responsibilities
There are 4 nodes in total 
- Supervisor agent: decides to route to revenue or expenditure agent
- Revenue agent: performs revenue analysis
- Expenditure agent: performs expenditure analysis
- Aggregator agent: consolidates information

To ensure that the revenue and expenditure agent has reliable information, I added a simulated RAG tool for the agents to gather the data, in this case I returned the full PDF data source in text format.


```
@tool
def get_pdf_data() -> dict:
    """
    Retrieves the complete PDF text data containing government budget information.
    
    Returns a dictionary where:
    - Keys are page numbers (integers)
    - Values are the text content of each page (strings)
    
    This data contains information about government revenue streams, expenditure,
    budget allocations, and specific fund details.
    """
    return pdf_text
```

The agent will invoke the tool to verify the information gathered, and perform it's analysis from there. 

### 3. Workflow building

Refer to the diagram below for the full workflow:
 ```
↓
[Supervisor Agent] ← Entry point, analyzes query
↓
[Conditional Edge] ← Routes based on supervisor_decision
↓↓
↓└─→ [Revenue Agent] (if "revenue" in decision)
↓
└──→ [Expenditure Agent] (if "expenditure" in decision)

Both agents can run in parallel ↓↓

[Aggregator Node] ← Combines revenue_info + expenditure_info
↓
[Final Answer Generation] ← Creates comprehensive response
↓
END
```

The choice to add a conditional edge was based on the complexity of the query, agent calling is expensive, and we would ideally want to invoke the necessary nodes within the graph to save memory, and time. 

We can trace the full flow of the workflow in an example output here

```
🎯 SUPERVISOR DECISION:
   Reasoning: The query asks about 'key government revenue streams' (revenue) and how the 'Budget for the Future Energy Fund' will be supported (expenditure). Therefore, both agents are needed.
   Agents needed: ['revenue', 'expenditure']

🔀 ROUTING DECISION: ['revenue', 'expenditure']
   Routing to: ['revenue_agent', 'expenditure_agent']

💸 EXPENDITURE AGENT: Calling tool to get PDF data...
   Tool: get_pdf_data

💰 REVENUE AGENT: Calling tool to get PDF data...
   Tool: get_pdf_data

💸 EXPENDITURE AGENT FINDINGS:
   Okay, I will provide a comprehensive expenditure analysis based on the provided PDF data, including specific figures and page citations.

**Overall Expenditure Trends**

*   **FY2023 (Revised):** Total Expenditure is revised to $106.9 billion (Page 6), which is $2.7 billion (2.6%) higher than the Es...

💰 REVENUE AGENT FINDINGS:
   Okay, here's a revenue analysis based on the provided PDF data, with specific figures and page citations.

**Overall Revenue Picture**
...
*   **FY2024 (Estimated):** Total Operating Revenue is projected to be $108.6 billion...

📊 AGGREGATOR - FINAL ANSWER GENERATED
   Length: 982 characters
```

The final output will be gathered from the aggregator node, in a consolidated form like this:

```
## Query: 
"""
What are the key government revenue streams, and how will the Budget for the Future Energy Fund be supported?"
"""

Based on the analysis:


**Key Government Revenue Streams:**

*   **Overall Revenue:** Total Operating Revenue was $104.3 billion in FY2023 (Revised) and is projected to be $108.6 billion in FY2024.
*   **Corporate Income Tax:** $28.38 billion (FY2023 Revised), $28.03 billion (FY2024 Estimated).
*   **Personal Income Tax:** $17.53 billion (FY2023 Revised), $18.07 billion (FY2024 Estimated).
*   **Goods and Services Tax (GST):** $16.36 billion (FY2023 Revised), $19.39 billion (FY2024 Estimated).
*   **Assets Taxes:** $5.92 billion (FY2023 Revised), $6.67 billion (FY2024 Estimated).
*   **Motor Vehicle Taxes:** $2.60 billion (FY2023 Revised), $2.84 billion (FY2024 Estimated).
*   **Vehicle Quota Premiums:** $4.66 billion (FY2023 Revised), $4.72 billion (FY2024 Estimated).

**Budget Support for the Future Energy Fund:**

*   The Future Energy Fund will receive a top-up of $5.0 billion in FY2024. This is part of the total $20.4 billion top-ups to Endowment and Trust Funds.
```

### Improvements

Limitations in the workflow now include:
- **Limited models** (gemini-2.5-flash instead of gpt-5):  using a reasoning model may work better for a more research heavy workflow
- **Limited tooling**: could possibly include more tools with access to internet or even internal data sources for more precise RAG methods
- **Feedback loop and human intervention**: Modern workflows may require a human in the loop to validate current research steps, at certain parts of the workflow (e.g the revenue agent), we can pause execution and let a human determine whether the information gathered is sufficient to proceed. There can also be a possibility of including a form of _critique agent_ to validate the outputs from the revenue and expenditure agent, but wasn't implemented due to interest of time

