import sqlite3
import os
from typing import Optional

DB_PATH = "customer_service_agent.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create customers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        shipping_address TEXT NOT NULL
    )
    """)
    
    # Create orders table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS orders (
        order_id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        status TEXT NOT NULL,
        shipping_address TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
    )
    """)
    
    # Insert mock customers if table is empty
    cursor.execute("SELECT COUNT(*) FROM customers")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
        INSERT INTO customers (customer_id, name, email, shipping_address)
        VALUES (?, ?, ?, ?)
        """, [
            ("cust_101", "Alice Smith", "alice@example.com", "123 Maple St, Seattle, WA 98101"),
            ("cust_102", "Bob Jones", "bob@example.com", "456 Oak St, Boston, MA 02108"),
            ("cust_103", "Charlie Brown", "charlie@example.com", "789 Pine St, San Francisco, CA 94103")
        ])
        
    # Insert mock orders if table is empty
    cursor.execute("SELECT COUNT(*) FROM orders")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("""
        INSERT INTO orders (order_id, customer_id, item_id, quantity, status, shipping_address)
        VALUES (?, ?, ?, ?, ?, ?)
        """, [
            (1024, "cust_101", "item_99", 2, "Shipped", "123 Maple St, Seattle, WA 98101"),
            (1025, "cust_102", "item_45", 1, "Processing", "456 Oak St, Boston, MA 02108")
        ])
        
    conn.commit()
    conn.close()

# Initialize on import
init_db()

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def db_get_customer(customer_id: str) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def db_get_order(order_id: int) -> Optional[dict]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def db_create_order(customer_id: str, item_id: str, quantity: int) -> int:
    # Fetch customer to get default shipping address
    customer = db_get_customer(customer_id)
    if not customer:
        raise ValueError(f"Customer {customer_id} does not exist.")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO orders (customer_id, item_id, quantity, status, shipping_address)
    VALUES (?, ?, ?, 'Pending', ?)
    """, (customer_id, item_id, quantity, customer["shipping_address"]))
    order_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return order_id

def db_update_order(order_id: int, shipping_address: Optional[str] = None, quantity: Optional[int] = None) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if order exists
    cursor.execute("SELECT 1 FROM orders WHERE order_id = ?", (order_id,))
    if not cursor.fetchone():
        conn.close()
        return False
        
    updates = []
    params = []
    if shipping_address is not None:
        updates.append("shipping_address = ?")
        params.append(shipping_address)
    if quantity is not None:
        updates.append("quantity = ?")
        params.append(quantity)
        
    if updates:
        params.append(order_id)
        query = f"UPDATE orders SET {', '.join(updates)} WHERE order_id = ?"
        cursor.execute(query, tuple(params))
        conn.commit()
        
    conn.close()
    return True
