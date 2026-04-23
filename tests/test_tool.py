# tests/test_tools.py

import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.tools.implementations.order_lookup import lookup_order
from core.tools.implementations.refund_processor import process_refund
from core.tools.implementations.ticket_creator import create_ticket

async def test_all_tools():
    print("Testing Tools...\n")

    # 1. Test Order Lookup (Looking up one of our fake generated orders!)
    print("1. Order Lookup:")
    order_res = await lookup_order("ORD-10005") # One of the orders generated in Day 3
    if order_res["ok"]:
        print(f" Found Order: {order_res['data']['product_name']} for ${order_res['data']['amount']}")
    else:
        print(f" Error: {order_res['error']}")

    # 2. Test Refund Processor
    print("\n2. Stripe Refund Processor:")
    refund_res = await process_refund("ORD-10005", 99.99, "Customer requested")
    if refund_res["ok"]:
        print(f" Refund Processed: ID {refund_res['data']['refund_id']} for ${refund_res['data']['amount']}")
    else:
        print(f" Error: {refund_res['error']}")

    # 3. Test Zendesk Ticket Creator
    print("\n3. Zendesk Ticket Creator:")
    ticket_res = await create_ticket("user-123", "conv-456", "Missing item in delivery", "high")
    if ticket_res["ok"]:
        print(f" Ticket Created: {ticket_res['data']['ticket_id']} - Priority: {ticket_res['data']['priority']}")
    else:
        print(f" Error: {ticket_res['error']}")

if __name__ == "__main__":
    asyncio.run(test_all_tools())