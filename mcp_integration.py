# mcp_integration.py
import requests
import json

class MCPClient:
    def __init__(self, mcp_server_url):
        self.server_url = mcp_server_url

    def get_capabilities(self):
        return requests.post(f"{self.server_url}/rpc", json={"method": "discover"}).json()

    def execute_tool(self, tool_name, params):
        payload = {
            "method": tool_name,
            "params": params
        }
        return requests.post(f"{self.server_url}/rpc", json=payload).json()

# Use as bridge for your agents if you have an MCP-compliant server exposing tools/data
