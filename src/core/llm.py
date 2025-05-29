from src.utils.utils import load_chat_model

# Initialize shared chat models for graph nodes
model = load_chat_model()
final_model = load_chat_model().with_config(tags=["final_node"])
