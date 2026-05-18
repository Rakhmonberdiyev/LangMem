import asyncio
from contextlib import AsyncExitStack
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

class MCPClient:
    """Nginx Gateway orqali ishlovchi Async MCP client"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.session = None
        self.available_tools = []
        self.exit_stack = None
        self.connected = False
    
    async def connect(self):
        if self.connected:
            return
        
        self.exit_stack = AsyncExitStack()
        
        # Nginx orqali bog'lanish
        read_stream, write_stream, get_session_id = await self.exit_stack.enter_async_context(
            streamablehttp_client(self.server_url)
        )
        
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        
        await self.session.initialize()
        
        tools_list = await self.session.list_tools()
        self.available_tools = tools_list.tools
        self.connected = True
        print(f"✅ Connected to: {self.server_url}")
    
    async def disconnect(self):
        if self.exit_stack:
            await self.exit_stack.aclose()
            self.connected = False
    
    async def call_tool(self, tool_name: str, arguments: dict):
        if not self.connected or not self.session:
            raise Exception("Not connected to MCP server")
        
        result = await self.session.call_tool(tool_name, arguments)
        return result.content[0].text
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()

async def main():
    # --- NGINX GATEWAY KONFIGURATSIYASI ---
    # Agar Nginx o'z kompyuteringizda bo'lsa 'localhost' ishlating
    # Agar boshqa serverda bo'lsa '172.28.23.100' ishlating
    GATEWAY_URL = "http://localhost:8080" # yoki "http://172.28.23.100"

    # Nginx location bloklariga mos URL'lar:
    urls = {
        "Deposit": f"{GATEWAY_URL}/deposit/mcp",
        "Credit": f"{GATEWAY_URL}/credit/mcp",
        "Pension": f"{GATEWAY_URL}/pension/mcp",
        "Card": f"{GATEWAY_URL}/card/mcp",
        "Admin": f"{GATEWAY_URL}/admin/mcp",
        "RealTime": f"{GATEWAY_URL}/real_time/mcp"
    }

    # Har bir servisni tekshirish
    for name, url in urls.items():
        try:
            async with MCPClient(url) as client:
                print(f"\n📦 {name} Server - Tools found: {len(client.available_tools)}")
                for tool in client.available_tools:
                    print(f"  • {tool.name}")
        except Exception as e:
            print(f"❌ {name} serverga ulanishda xato: {e}")

if __name__ == "__main__":
    asyncio.run(main())