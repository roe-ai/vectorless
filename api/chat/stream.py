import sys
import os
import json
import time
import asyncio
from http.server import BaseHTTPRequestHandler

# Add the backend directory to the Python path before importing
backend_path = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from models import ChatRequest
from llm_service import LLMService


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            # Set CORS headers for streaming
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Read request body
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode("utf-8"))

            # Parse request using ChatRequest model
            request = ChatRequest(**request_data)

            # Initialize LLM service
            llm_service = LLMService()

            # Process the chat request and stream response
            asyncio.run(self._process_chat_request(request, llm_service))

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            self.wfile.write(f"data: {json.dumps(error_data)}\n\n".encode())
            print(f"❌ Error in chat handler: {str(e)}")

    async def _process_chat_request(self, request, llm_service):
        """Process chat request with streaming response"""
        start_time = time.time()
        print(f"🌊 Streaming chat request started")
        print(f"📝 Question: {request.question}")
        print(f"📊 Received {len(request.documents)} documents")

        try:
            total_cost = 0.0

            # Convert DocumentData to the format expected by LLMService
            documents_dict = []
            for doc in request.documents:
                pages_dict = []
                for page in doc.pages:
                    pages_dict.append(
                        {"page_number": page.page_number, "text": page.text}
                    )

                documents_dict.append(
                    {
                        "id": doc.id,
                        "filename": doc.filename,
                        "pages": pages_dict,
                        "total_pages": doc.total_pages,
                    }
                )

            # Step 1: Select relevant documents
            step1_start = time.time()
            doc_selection_status = {
                "type": "status",
                "step": "document_selection",
                "message": "Finding relevant documents...",
                "step_number": 1,
                "total_steps": 3,
            }
            data = f"data: {json.dumps(doc_selection_status)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            print("⏱️ Step 1: Starting document selection...")
            selected_docs, step1_cost = await llm_service.select_documents(
                request.description,
                documents_dict,
                request.question,
                request.chat_history,
                request.model,
            )
            total_cost += step1_cost
            step1_time = time.time() - step1_start
            msg = f"✅ Step 1: Document selection completed in {step1_time:.2f}s"
            print(msg)

            # Send completion status for document selection
            doc_selection_complete = {
                "type": "step_complete",
                "step": "document_selection",
                "selected_documents": [
                    {"id": doc["id"], "filename": doc["filename"]}
                    for doc in selected_docs
                ],
                "cost": step1_cost,
                "time_taken": step1_time,
            }
            data = f"data: {json.dumps(doc_selection_complete)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            # Step 2: Find relevant pages
            step2_start = time.time()
            page_selection_status = {
                "type": "status",
                "step": "page_selection",
                "message": "Finding relevant pages in selected documents...",
                "step_number": 2,
                "total_steps": 3,
            }
            data = f"data: {json.dumps(page_selection_status)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            print("⏱️ Step 2: Starting page selection...")

            async def process_document(doc):
                return await llm_service.find_relevant_pages(
                    doc["pages"],
                    request.question,
                    doc["filename"],
                    request.chat_history,
                    request.model,
                )

            # Create tasks for all documents
            doc_tasks = [process_document(doc) for doc in selected_docs]

            # Wait for all documents to complete
            doc_results = await asyncio.gather(*doc_tasks)

            # Combine results
            all_relevant_pages = []
            step2_cost = 0.0
            for doc_relevant_pages, doc_cost in doc_results:
                all_relevant_pages.extend(doc_relevant_pages)
                step2_cost += doc_cost

            relevant_pages = all_relevant_pages
            total_cost += step2_cost
            step2_time = time.time() - step2_start
            msg = f"✅ Step 2: Page selection completed in {step2_time:.2f}s"
            print(msg)

            # Send completion status for page selection
            page_selection_complete = {
                "type": "step_complete",
                "step": "page_selection",
                "relevant_pages_count": len(relevant_pages),
                "cost": step2_cost,
                "time_taken": step2_time,
            }
            data = f"data: {json.dumps(page_selection_complete)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            # Step 3: Generate answer
            step3_start = time.time()
            answer_generation_status = {
                "type": "status",
                "step": "answer_generation",
                "message": "Generating comprehensive answer...",
                "step_number": 3,
                "total_steps": 3,
            }
            data = f"data: {json.dumps(answer_generation_status)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            print("⏱️ Step 3: Starting answer generation...")

            # Stream the answer generation
            async for chunk in llm_service.generate_answer_stream(
                relevant_pages, request.question, request.chat_history, request.model
            ):
                if chunk.get("type") == "content":
                    content_data = {
                        "type": "content",
                        "content": chunk["content"],
                    }
                    data = f"data: {json.dumps(content_data)}\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
                elif chunk.get("type") == "cost":
                    total_cost += chunk["cost"]

            step3_time = time.time() - step3_start
            msg = f"✅ Step 3: Answer generation completed in {step3_time:.2f}s"
            print(msg)

            # Send final completion
            total_time = time.time() - start_time
            completion_data = {
                "type": "complete",
                "timing_breakdown": {
                    "document_selection": step1_time,
                    "page_detection": step2_time,
                    "answer_generation": step3_time,
                    "total_time": total_time,
                },
                "cost_breakdown": {
                    "document_selection": step1_cost,
                    "page_detection": step2_cost,
                    "answer_generation": total_cost - step1_cost - step2_cost,
                    "total_cost": total_cost,
                },
            }
            data = f"data: {json.dumps(completion_data)}\n\n"
            self.wfile.write(data.encode())
            self.wfile.flush()

            cost_msg = f"🎉 Request completed in {total_time:.2f}s, total cost: ${total_cost:.4f}"
            print(cost_msg)

        except Exception as e:
            error_data = {"type": "error", "error": str(e)}
            data = f"data: {json.dumps(error_data)}\n\n"
            self.wfile.write(data.encode())
            print(f"❌ Error in stream_response: {str(e)}")

    def do_OPTIONS(self):
        # Handle CORS preflight
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        # Add GET method for testing
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.end_headers()

        response_data = {
            "message": "Chat stream endpoint",
            "method": "POST",
            "description": "Use POST method to send chat requests",
        }

        self.wfile.write(json.dumps(response_data).encode())
