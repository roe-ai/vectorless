import asyncio
import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
import json

load_dotenv()


class LLMService:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key or api_key == "your_openai_api_key_here":
            print("⚠️  OpenAI API key not set. LLM features will be disabled.")
            self.client = None
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        self.anthropic_client = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
        
        self.model = "gpt-5-mini"

        self.pricing = {
            "gpt-5": {"input": 1.25, "output": 10.0},
            "gpt-5-mini": {"input": 0.25, "output": 2.0},
            "claude-haiku-4.5": {"input": 0.80, "output": 4.0},
        }

        self.model_to_provider = {
            "gpt-5-mini": "openai",
            "gpt-5": "openai",
            "claude-haiku-4.5": "anthropic",
        }

        self.model_ids = {
            "claude-haiku-4.5": "claude-haiku-4-5-20251001",
        }

    def get_default_model(self) -> Optional[str]:
        if self.client:
            return "gpt-5-mini"
        if self.anthropic_client:
            return "claude-haiku-4.5"
        return None

    def list_available_models(self) -> List[Dict[str, Any]]:
        models = []
        default_model = self.get_default_model()
        
        if self.client:
            models.extend([
                {"id": "gpt-5-mini", "label": "GPT-5 Mini", "description": "OpenAI • Faster responses, lower cost", "provider": "openai", "pricing": self.pricing["gpt-5-mini"], "is_default": "gpt-5-mini" == default_model},
                {"id": "gpt-5", "label": "GPT-5", "description": "OpenAI • Highest quality, highest cost", "provider": "openai", "pricing": self.pricing["gpt-5"], "is_default": "gpt-5" == default_model},
            ])
        
        if self.anthropic_client:
            models.append({"id": "claude-haiku-4.5", "label": "Claude 4.5 Haiku", "description": "Anthropic • Balanced quality and price", "provider": "anthropic", "pricing": self.pricing["claude-haiku-4.5"], "is_default": "claude-haiku-4.5" == default_model})
        
        return models

    def _resolve_model(self, requested_model: Optional[str]) -> str:
        if requested_model:
            provider = self.model_to_provider.get(requested_model)
            if provider == "openai" and self.client:
                return requested_model
            elif provider == "anthropic" and self.anthropic_client:
                return requested_model
        default = self.get_default_model()
        if default:
            return default
        raise RuntimeError("No LLM providers configured")

    def calculate_cost(self, usage_data, model="gpt-5-mini"):
        print(usage_data)
        """Calculate cost based on token usage"""
        if not usage_data or model not in self.pricing:
            return 0.0

        # Handle both OpenAI and Anthropic formats
        if hasattr(usage_data, "prompt_tokens"):
            input_tokens = usage_data.prompt_tokens
            output_tokens = usage_data.completion_tokens
        elif hasattr(usage_data, "input_tokens"):
            input_tokens = usage_data.input_tokens
            output_tokens = usage_data.output_tokens
        else:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * self.pricing[model]["input"]
        output_cost = (output_tokens / 1_000_000) * self.pricing[model]["output"]

        return input_cost + output_cost

    async def select_documents(
        self,
        description: str,
        documents: List[Dict[str, Any]],
        question: str,
        chat_history: List[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        """
        Select relevant documents based on description, question, and chat history
        """
        resolved_model = self._resolve_model(model)

        doc_summaries = []
        for doc in documents:
            doc_summaries.append(
                {
                    "id": doc["id"],
                    "filename": doc["filename"],
                    "total_pages": doc["total_pages"],
                    "first_page_preview": (doc["pages"][0]["text"][:500] + "..."),
                }
            )

        # Format chat history
        history_context = ""
        if chat_history:
            history_context = "\n\nChat History:\n"
            for msg in chat_history:
                if hasattr(msg, "role"):
                    role = msg.role
                    content = msg.content
                else:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                history_context += f"{role.capitalize()}: {content}\n"

        prompt = f"""
            Based on the following document collection description, chat history, 
            and current question, select which documents are most likely to 
            contain the answer.

            <Document Collection Description>
            {description}
            <Document Collection Description>

            <Available Documents>
            {json.dumps(doc_summaries, indent=2)}
            <Available Documents>

            <Chat History>
            {history_context}
            <Chat History>

            <Current Question>
            {question}
            <Current Question>

            Return a JSON array of document IDs (numbers) that are most relevant to 
            the current question and conversation context.
            Only return the JSON array, no other text.
            Example: [1, 3, 5]
            """

        try:
            if self.model_to_provider[resolved_model] == "openai":
                response = await self.client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.choices[0].message.content
                usage = response.usage
            else:  # anthropic
                response = await self.anthropic_client.messages.create(
                    model=self.model_ids.get(resolved_model, resolved_model),
                    max_tokens=4096,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = "".join(block.text for block in response.content if hasattr(block, "text"))
                usage = response.usage

            selected_ids = json.loads(content)
            cost = self.calculate_cost(usage, resolved_model)

            # Return full document objects for selected IDs
            selected_docs = []
            for doc in documents:
                if doc["id"] in selected_ids:
                    selected_docs.append(doc)

            return selected_docs, cost

        except Exception as e:
            print(f"Error in document selection: {e}")
            # Fallback: return all documents
            return documents, 0.0

    async def find_relevant_pages(
        self,
        pages: List[Dict[str, Any]],
        question: str,
        filename: str,
        chat_history: List[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], float]:
        print("find_relevant_pages")
        print(filename)
        """Find relevant pages by processing 20 pages at a time in parallel"""

        resolved_model = self._resolve_model(model)

        # Create chunks of 20 pages
        chunks = []
        for i in range(0, len(pages), 20):
            chunk = pages[i : i + 20]
            chunks.append(chunk)

        # Process all chunks in parallel
        chunk_tasks = []
        for chunk_index, chunk in enumerate(chunks):
            task = self._process_page_chunk(
                chunk, question, filename, chunk_index, chat_history, resolved_model
            )
            chunk_tasks.append(task)

        # Wait for all chunks to complete
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        # Combine results from all chunks
        relevant_pages = []
        total_cost = 0.0
        for result in chunk_results:
            if isinstance(result, Exception):
                print(f"Error in chunk processing: {result}")
                continue
            if isinstance(result, tuple) and len(result) == 2:
                pages, cost = result
                relevant_pages.extend(pages)
                total_cost += cost
            elif isinstance(result, list):
                # Fallback for old format
                relevant_pages.extend(result)

        return relevant_pages, total_cost

    async def _process_page_chunk(
        self,
        chunk: List[Dict[str, Any]],
        question: str,
        filename: str,
        chunk_index: int,
        chat_history: List[Dict[str, Any]] = None,
        model: str = "",
    ) -> tuple[List[Dict[str, Any]], float]:
        """Process a single chunk of pages"""
        import time

        chunk_start = time.time()
        print(f"    🔍 Processing chunk {chunk_index + 1} with {len(chunk)} pages...")

        # Prepare content for LLM
        pages_content = []
        for page in chunk:
            # Defensive check for required fields
            if "page_number" not in page:
                print(f"Warning: page missing 'page_number': {page.keys()}")
                continue
            if "text" not in page:
                print(f"Warning: page missing 'text': {page.keys()}")
                continue

            pages_content.append(
                {
                    "page_number": page["page_number"],
                    "page_content": (page["text"]),
                }
            )

        # Format chat history for context
        history_context = ""
        if chat_history:
            history_context = "\n\nRecent Chat History:\n"
            for msg in chat_history:
                if hasattr(msg, "role"):
                    role = msg.role
                    content = msg.content
                else:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                history_context += f"{role.capitalize()}: {content}...\n"

        prompt = f"""
            Analyze the following pages from document "{filename}" and determine 
            which pages are relevant to the current question, considering the conversation context. 
            Return empty array if no pages are relevant.
            
            <Chat History>
            {history_context}
            <Chat History>
            
            <Current Question>
            {question}
            <Current Question>

            <Document Page Content>
            {json.dumps(pages_content, indent=2)}
            <Document Page Content>

            Return a JSON array of page numbers relevant to the current question
            Only return the JSON array, no other text.
            Example: [1, 3, 5]
            """

        try:
            if self.model_to_provider[model] == "openai":
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = response.choices[0].message.content
                usage = response.usage
            else:  # anthropic
                response = await self.anthropic_client.messages.create(
                    model=self.model_ids.get(model, model),
                    max_tokens=4096,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                response_text = "".join(block.text for block in response.content if hasattr(block, "text"))
                usage = response.usage

            relevant_page_numbers = json.loads(response_text)
            cost = self.calculate_cost(usage, model=model)

            # Add full page data for relevant pages
            relevant_pages = []
            for page in chunk:
                if "page_number" not in page:
                    continue
                if page["page_number"] in relevant_page_numbers:
                    page_with_source = page.copy()
                    page_with_source["source_document"] = filename
                    relevant_pages.append(page_with_source)

            chunk_time = time.time() - chunk_start
            print(
                f"    ✅ Chunk {chunk_index + 1} completed in {chunk_time:.2f}s, found {len(relevant_pages)} relevant pages"
            )
            return relevant_pages, cost

        except Exception as e:
            chunk_time = time.time() - chunk_start
            print(f"    ❌ Chunk {chunk_index + 1} failed in {chunk_time:.2f}s: {e}")
            # Fallback: include first page of chunk
            if chunk:
                first_page = chunk[0].copy()
                first_page["source_document"] = filename
                return [first_page], 0.0
            return [], 0.0

    async def generate_answer_stream(
        self,
        relevant_pages: List[Dict[str, Any]],
        question: str,
        chat_history: List[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ):
        """Generate final answer using all relevant pages with streaming"""

        if not relevant_pages:
            yield {
                "type": "content",
                "content": "I couldn't find any relevant information to answer your question.",
            }
            yield {"type": "cost", "cost": 0.0}
            return

        resolved_model = self._resolve_model(model)

        # Format chat history for conversational context
        history_context = ""
        if chat_history:
            history_context = "\n\nConversation History:\n"
            for msg in chat_history:  # Include last 4 messages for context
                if hasattr(msg, "role"):
                    role = msg.role
                    content = msg.content
                else:
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                history_context += f"{role.capitalize()}: {content}\n"
        print(relevant_pages)
        prompt = f"""
            Based on the following chat history context, PDF document context, and current question, answer the question. 
            Provide answer and cite which documents and pages you're referencing.

            IMPORTANT: When referencing specific pages, use this special format:
            $PAGE_START{{filename}}:{{page_numbers}}$PAGE_END

            Examples:
            - For single page: $PAGE_STARTreport.pdf:5$PAGE_END
            - For multiple pages: $PAGE_STARTanalysis.pdf:2,7,12$PAGE_END 
            - For page range: $PAGE_STARTmanual.pdf:15-18$PAGE_END

            <Chat History>
            {history_context}
            <Chat History>
            
            <Current Question>
            {question}
            <Current Question>

            <Document Page Content>
            {json.dumps(relevant_pages)}
            <Document Page Content>

            Please provide answer based on the information in the documents and use the special page reference format when citing specific pages.
            No need to mention the chat history in the answer, just focus on the current question.
            """

        try:
            if self.model_to_provider[resolved_model] == "openai":
                stream = await self.client.chat.completions.create(
                    model=resolved_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    stream_options={"include_usage": True},
                )

                async for chunk in stream:
                    if chunk.usage:
                        yield {
                            "type": "cost",
                            "cost": self.calculate_cost(chunk.usage, model=resolved_model),
                        }
                    if len(chunk.choices) > 0:
                        if chunk.choices[0].delta.content is not None:
                            yield {
                                "type": "content",
                                "content": chunk.choices[0].delta.content,
                            }

            else:  # anthropic
                response = await self.anthropic_client.messages.create(
                    model=self.model_ids.get(resolved_model, resolved_model),
                    max_tokens=4096,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )

                for block in response.content:
                    if hasattr(block, "text") and block.text:
                        yield {
                            "type": "content",
                            "content": block.text,
                        }

                if response.usage:
                    yield {
                        "type": "cost",
                        "cost": self.calculate_cost(response.usage, model=resolved_model),
                    }

        except Exception as e:
            yield {"type": "content", "content": f"Error generating answer: {str(e)}"}
            yield {"type": "cost", "cost": 0.0}
