from src.langgraphagenticai.states.states import State,Route1
from src.langgraphagenticai.promts import router_template
from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced"
)


class Bot1:
    def __init__(self, model):
        self.llm = model

    def routing(self, state: State):
        router_chain = router_template | self.llm.with_structured_output(Route1)

        result = router_chain.invoke({
            "user_input": state["input"],
            "history": state.get("messages", []),
            "context": state.get("output", "")
        })

        state["action"] = result.action
        state["category"] = result.category
        return state

    def tavily_search_node(self, state: State) -> State:
        query = state["input"]
        results = tavily_tool.invoke(query)

        context = "\n".join(
            f"- {r.get('title','')}\n  {r.get('content','')}\n  Source: {r.get('url','')}"
            for r in results
        )

        state["output"] = context
        state.setdefault("messages", []).append({
            "role": "tool",
            "name": "tavily_search",
            "content": context
        })

        return state
    

    # def vector_stores(self,state:State):
