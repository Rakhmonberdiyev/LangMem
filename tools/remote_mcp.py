"""Proxy FastMCP servers for the Nginx-gated remote MCP services."""

from fastmcp.server import create_proxy

GATEWAY_URL = "http://localhost:8080"

deposit_mcp  = create_proxy(f"{GATEWAY_URL}/deposit/mcp",   name="Deposit")
credit_mcp   = create_proxy(f"{GATEWAY_URL}/credit/mcp",    name="Credit")
pension_mcp  = create_proxy(f"{GATEWAY_URL}/pension/mcp",   name="Pension")
card_mcp     = create_proxy(f"{GATEWAY_URL}/card/mcp",      name="Card")
admin_mcp    = create_proxy(f"{GATEWAY_URL}/admin/mcp",     name="Admin")
realtime_mcp = create_proxy(f"{GATEWAY_URL}/real_time/mcp", name="RealTime")
