from src.agents.states.states import State, Route1
from src.agents.prompts import router_template
from dotenv import load_dotenv
load_dotenv()
from src.services.tavily_service import tavily_search as _tavily_search


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
        context = _tavily_search(query)

        state["output"] = context
        state.setdefault("messages", []).append({
            "role": "tool",
            "name": "tavily_search",
            "content": context
        })

        return state
