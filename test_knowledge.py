import asyncio
from core.agents.knowledge_agent import KnowledgeAgent

async def main():
    agent = KnowledgeAgent()
    res = await agent.run('tell me about return policy of bans', 'general_inquiry', '')
    print('RESPONSE:', repr(res.output.get('response')))

asyncio.run(main())
