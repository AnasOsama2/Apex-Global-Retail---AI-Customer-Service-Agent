from typing import Optional
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from app.tools.db import db_create_order, db_update_order, db_get_order

class CreateOrderInput(BaseModel):
    customer_id: str = Field(description="The unique identifier for the customer.")
    item_id: str = Field(description="SKU or product identifier.")
    quantity: int = Field(description="Number of units ordered.", gt=0)
    shipping_address: Optional[str] = Field(default=None, description="Optional custom shipping address for the order. If not provided, the customer's default address will be used.")

class UpdateOrderInput(BaseModel):
    order_id: int = Field(description="The unique identifier for the order to update.")
    shipping_address: Optional[str] = Field(default=None, description="The new shipping address for the order.")
    quantity: Optional[int] = Field(default=None, description="The new quantity for the order.", gt=0)

class GetOrderStatusInput(BaseModel):
    order_id: int = Field(description="The unique identifier for the order to retrieve status.")

@tool("create_order", args_schema=CreateOrderInput)
def create_order(customer_id: str, item_id: str, quantity: int, shipping_address: Optional[str] = None) -> str:
    """Executes an order placement in the database. Use this tool when the customer explicitly wants to buy an item."""
    try:
        order_id = db_create_order(customer_id, item_id, quantity, shipping_address)
        return f"Success: Order #{order_id} created for customer {customer_id}."
    except Exception as e:
        return f"Error: Failed to create order. {str(e)}"

@tool("update_order", args_schema=UpdateOrderInput)
def update_order(order_id: int, shipping_address: Optional[str] = None, quantity: Optional[int] = None) -> str:
    """Updates an existing order's details (shipping address and/or quantity) in the database. Use this tool when the customer wants to change their shipping address or quantity of an order."""
    try:
        success = db_update_order(order_id, shipping_address, quantity)
        if success:
            return f"Success: Order #{order_id} updated."
        else:
            return f"Error: Order #{order_id} not found."
    except Exception as e:
        return f"Error: Failed to update order. {str(e)}"

@tool("get_order_status", args_schema=GetOrderStatusInput)
def get_order_status(order_id: int) -> str:
    """Retrieves the current status of an order. Use this tool when the customer wants to track or check the status of their order."""
    try:
        order = db_get_order(order_id)
        if order:
            return f"Success: Order #{order_id} status is '{order['status']}', shipping to {order['shipping_address']}."
        else:
            return f"Error: Order #{order_id} not found."
    except Exception as e:
        return f"Error: Failed to retrieve order status. {str(e)}"
