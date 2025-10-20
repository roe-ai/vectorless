'use client';

import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Send, Upload, X, FileText, Eye, EyeOff, Settings, Plus, Trash2, Edit2, Check, RotateCcw } from 'lucide-react';
import config from '../../config';
import ChunkedUploader from '../utils/chunkedUpload';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  progress?: {
    status: string;
    step: number;
    total: number;
    stepCost?: number;
    stepTime?: number;
  };
  metadata?: {
    selectedDocuments?: Array<{id: number, filename: string}>;
    relevantPagesCount?: number;
    timing?: {
      document_selection?: number;
      page_detection?: number;
      answer_generation?: number;
      total_time?: number;
    };
    costs?: {
      document_selection?: number;
      page_detection?: number;
      answer_generation?: number;
      total_cost?: number;
    };
    model?: string;
  };
}

interface DocumentData {
  id: number;
  filename: string;
  pages: Array<{
    page_number: number;
    text: string;
  }>;
  total_pages: number;
}

interface StatelessChatSectionProps {
  documents: DocumentData[];
  description: string;
  onReset: () => void;
  onUpdateDocuments: (documents: DocumentData[]) => void;
  onUpdateDescription: (description: string) => void;
}

interface AvailableModel {
  id: string;
  label: string;
  description?: string;
  provider: string;
  is_default?: boolean;
}

export default function StatelessChatSection({ 
  documents, 
  description, 
  onReset, 
  onUpdateDocuments, 
  onUpdateDescription 
}: StatelessChatSectionProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentQuestion, setCurrentQuestion] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showDocuments, setShowDocuments] = useState(true);
  const [selectedModel, setSelectedModel] = useState<string>('');
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([]);
  const [isEditingDescription, setIsEditingDescription] = useState(false);
  const [editedDescription, setEditedDescription] = useState('');
  const [isUploadingFiles, setIsUploadingFiles] = useState(false);
  const [selectedNewFiles, setSelectedNewFiles] = useState<File[]>([]);
  const [showUploadSection, setShowUploadSection] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadProgress, setUploadProgress] = useState<{
    percentComplete: number;
    processedFiles: number;
    totalFiles: number;
    currentChunk: number;
    totalChunks: number;
    isChunked: boolean;
  } | null>(null);
  const [totalSessionCost, setTotalSessionCost] = useState<number>(0);
  const [selectedPageContent, setSelectedPageContent] = useState<{
    content: string;
    pageNumber: number;
    filename: string;
  } | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch(`${config.apiBaseUrl}/models`);
        if (response.ok) {
          const data = await response.json();
          const models = data.models || [];
          setAvailableModels(models);
          if (!selectedModel && data.default_model) {
            setSelectedModel(data.default_model);
          }
        }
      } catch (error) {
        console.error('Failed to fetch models:', error);
      }
    };
    fetchModels();
  }, []);

  const handleSendMessage = async () => {
    if (!currentQuestion.trim() || isLoading || !selectedModel) return;

    const question = currentQuestion;
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: question,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setCurrentQuestion('');
    setIsLoading(true);

    // Create assistant message placeholder for streaming
    const assistantMessageId = (Date.now() + 1).toString();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true,
      progress: {
        status: 'Starting...',
        step: 0,
        total: 3,
      },
      metadata: {
        selectedDocuments: [],
        relevantPagesCount: 0,
      },
    };
    
    setMessages(prev => [...prev, assistantMessage]);

    try {
      const response = await fetch(`${config.apiBaseUrl}/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question: question,
          documents: documents,
          description: description,
          model: selectedModel,
          chat_history: messages.map(msg => ({
            role: msg.role,
            content: msg.content,
            timestamp: msg.timestamp.toISOString()
          }))
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to get response');
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('No reader available');
      }

      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'status') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        progress: {
                          status: data.message,
                          step: data.step_number,
                          total: data.total_steps,
                        }
                      }
                    : msg
                ));
              } else if (data.type === 'step_complete') {
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        progress: {
                          ...msg.progress!,
                          stepCost: data.cost,
                          stepTime: data.time_taken,
                        },
                        metadata: {
                          ...msg.metadata,
                          ...(data.step === 'document_selection' && {
                            selectedDocuments: data.selected_documents,
                          }),
                          ...(data.step === 'page_selection' && {
                            relevantPagesCount: data.relevant_pages_count,
                          }),
                        }
                      }
                    : msg
                ));
              } else if (data.type === 'content') {
                accumulatedContent += data.content;
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? { ...msg, content: accumulatedContent }
                    : msg
                ));
              } else if (data.type === 'complete') {
                if (data.cost_breakdown?.total_cost) {
                  setTotalSessionCost(prev => prev + data.cost_breakdown.total_cost);
                }
                
                setMessages(prev => prev.map(msg => 
                  msg.id === assistantMessageId 
                    ? {
                        ...msg,
                        isStreaming: false,
                        progress: undefined,
                        metadata: {
                          ...msg.metadata,
                          timing: data.timing_breakdown,
                          costs: data.cost_breakdown,
                          model: selectedModel
                        }
                      }
                    : msg
                ));
                
                // Reset loading state when streaming is complete
                setIsLoading(false);
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (parseError) {
              console.warn('Failed to parse SSE data:', line);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? {
              ...msg,
              content: 'Sorry, I encountered an error while processing your question. Please try again.',
              isStreaming: false,
              progress: undefined
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleEditDescription = () => {
    setEditedDescription(description);
    setIsEditingDescription(true);
  };

  const handleSaveDescription = () => {
    if (!editedDescription.trim()) {
      return;
    }
    onUpdateDescription(editedDescription);
    setIsEditingDescription(false);
  };

  const handleCancelEdit = () => {
    setIsEditingDescription(false);
    setEditedDescription('');
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || []);
    
    if (files.length > 100) {
      setUploadError('Maximum 100 files allowed');
      return;
    }

    const invalidFiles = files.filter(file => !file.name.endsWith('.pdf'));
    if (invalidFiles.length > 0) {
      setUploadError('Only PDF files are allowed');
      return;
    }

    const currentCount = documents.length;
    if (currentCount + files.length > 100) {
      setUploadError(`Adding ${files.length} files would exceed the 100 document limit. Current: ${currentCount}`);
      return;
    }

    const existingFilenames = documents?.map(doc => doc?.filename).filter(Boolean) || [];
    const duplicateFiles = files.filter(file => existingFilenames.includes(file.name));
    if (duplicateFiles.length > 0) {
      setUploadError(`Files already exist: ${duplicateFiles.map(f => f.name).join(', ')}`);
      return;
    }

    setSelectedNewFiles(files);
    setUploadError('');
  };

  const removeNewFile = (index: number) => {
    setSelectedNewFiles(files => files.filter((_, i) => i !== index));
  };

  const handleUploadNewFiles = async () => {
    if (selectedNewFiles.length === 0) {
      setUploadError('Please select at least one PDF file');
      return;
    }

    setIsUploadingFiles(true);
    setUploadError('');
    setUploadProgress(null);

    try {
      // Check if chunking will be needed
      const needsChunking = ChunkedUploader.needsChunking(selectedNewFiles);
      
      // Use the new ChunkedUploader
      const result = await ChunkedUploader.upload({
        endpoint: `${config.apiBaseUrl}/upload`,
        files: selectedNewFiles,
        description: description,
        onProgress: (progress) => {
          setUploadProgress({
            ...progress,
            isChunked: needsChunking
          });
          console.log(`Upload progress: ${progress.percentComplete}% (${progress.processedFiles}/${progress.totalFiles} files, chunk ${progress.currentChunk}/${progress.totalChunks})`);
        },
        onChunkComplete: (chunkIndex, totalChunks) => {
          console.log(`Completed chunk ${chunkIndex + 1} of ${totalChunks}`);
        }
      });

      if (!result.success) {
        throw new Error(result.error || 'Upload failed');
      }

      console.log('Upload response:', result);
      
      // Validate the response format
      if (!result.documents || !Array.isArray(result.documents)) {
        console.error('Invalid response format. Expected documents array, got:', result);
        throw new Error(`Invalid response format from server. Expected documents array, got: ${JSON.stringify(result, null, 2)}`);
      }
      
      // Get the highest existing ID
      const maxId = documents && documents.length > 0 ? Math.max(...documents.map(d => d?.id || 0).filter(id => id > 0)) : 0;
      
      // Update IDs for new documents
      const newDocuments = result.documents.map((doc: DocumentData, index: number) => ({
        ...doc,
        id: maxId + index + 1
      }));
      
      // Add to existing documents
      onUpdateDocuments([...documents, ...newDocuments]);
      
      setSelectedNewFiles([]);
      setShowUploadSection(false);
      
      // Show success message with chunking info
      const uploadMessage = result.chunked 
        ? `Successfully uploaded ${result.totalFiles} files using ${ChunkedUploader.estimateChunks(selectedNewFiles)} chunks`
        : `Successfully uploaded ${result.totalFiles} files`;
      
      console.log(uploadMessage);
      
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error instanceof Error ? error.message : 'Upload failed');
    } finally {
      setIsUploadingFiles(false);
      setUploadProgress(null);
    }
  };

  const handleCancelUpload = () => {
    setSelectedNewFiles([]);
    setShowUploadSection(false);
    setUploadError('');
  };

  const handleDeleteDocument = (documentId: number) => {
    // Confirm deletion
    if (confirm('Are you sure you want to delete this document?')) {
      const updatedDocuments = documents.filter(doc => doc.id !== documentId);
      onUpdateDocuments(updatedDocuments);
      
      // If no documents left, clear description as well
      if (updatedDocuments.length === 0) {
        onUpdateDescription('');
      }
    }
  };

  // Function to process page references in content before markdown
  const processPageReferences = (content: string): string => {
    if (!content.includes('$PAGE_START')) {
      return content;
    }
    
    // Parse special markers: $PAGE_STARTfilename:pages$PAGE_END
    const pageRefRegex = /\$PAGE_START([^:]+):([^\$]+)\$PAGE_END/g;
    
    // Replace markers with markdown links
    const processedContent = content.replace(pageRefRegex, (match, filename, pageSpec) => {
      const cleanFilename = filename.trim();
      const cleanPageSpec = pageSpec.trim();
      
      // Convert to markdown link format with special data attributes
      return `[${cleanFilename} (Page ${cleanPageSpec})](javascript:void(0) "data-page-ref=${cleanFilename}:${cleanPageSpec}")`;
    });
    
    return processedContent;
  };

  // Helper function to parse page specifications like "5", "2,7,12", "15-18"
  const parsePageSpecification = (pageSpec: string): number[] => {
    const pages: number[] = [];
    
    // Split by commas for multiple pages/ranges
    const parts = pageSpec.split(',');
    
    parts.forEach(part => {
      part = part.trim();
      
      if (part.includes('-')) {
        // Handle range like "15-18"
        const [start, end] = part.split('-').map(p => parseInt(p.trim()));
        if (start && end && start <= end) {
          for (let i = start; i <= end; i++) {
            pages.push(i);
          }
        }
      } else {
        // Handle single page
        const pageNum = parseInt(part);
        if (pageNum) {
          pages.push(pageNum);
        }
      }
    });
    
    return pages;
  };

  const handlePageReference = (filename: string, pageNumber: number) => {
    // Find the document and page content
    const document = documents.find(doc => doc.filename === filename);
    if (!document) {
      console.error('Document not found:', filename);
      return;
    }
    
    const page = document.pages.find(p => p.page_number === pageNumber);
    if (!page) {
      console.error('Page not found:', pageNumber, 'in', filename);
      return;
    }
    
    setSelectedPageContent({
      content: page.text,
      pageNumber: pageNumber,
      filename: filename
    });
  };

  return (
    <div className="bg-white rounded-lg shadow-lg flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 p-4">
        <div className="flex justify-between items-center mb-3">
          <div className="flex-1">
            <h2 className="text-xl font-semibold text-gray-800">Chat with your documents</h2>
            <div className="flex items-center space-x-2 mt-1">
              <p className="text-sm text-gray-600">
                {documents.length} document(s) {documents.length > 0 ? '•' : ''}
              </p>
              {isEditingDescription ? (
                <div className="flex items-center space-x-2 flex-1">
                  <input
                    type="text"
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    className="text-sm text-gray-600 border border-gray-300 rounded px-2 py-1 flex-1"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        handleSaveDescription();
                      } else if (e.key === 'Escape') {
                        handleCancelEdit();
                      }
                    }}
                    autoFocus
                  />
                  <button
                    onClick={handleSaveDescription}
                    className="text-xs bg-blue-600 text-white px-2 py-1 rounded hover:bg-blue-700"
                  >
                    Save
                  </button>
                  <button
                    onClick={handleCancelEdit}
                    className="text-xs bg-gray-500 text-white px-2 py-1 rounded hover:bg-gray-600"
                  >
                    Cancel
                  </button>
                </div>
                              ) : documents.length > 0 ? (
                  <div className="flex items-center space-x-2">
                    <p className="text-sm text-gray-600">{description}</p>
                    <button
                      onClick={handleEditDescription}
                      className="text-xs text-blue-600 hover:text-blue-800"
                      title="Edit description"
                    >
                      Update document usage guide ✏️ 
                    </button>
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 italic">Upload your first PDF document to get started</p>
                )}
            </div>
          </div>
          <div className="flex space-x-2">
            <button
              onClick={() => setShowUploadSection(!showUploadSection)}
              className={documents.length === 0 
                ? "bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded text-sm font-medium"
                : "text-gray-500 hover:text-gray-700 px-3 py-1 rounded border border-gray-300 hover:border-gray-400 text-sm"
              }
            >
              {documents.length === 0 ? "Add Your First Document" : "Add Files"}
            </button>
            <button
              onClick={onReset}
              className="text-gray-500 hover:text-gray-700 px-3 py-1 rounded border border-gray-300 hover:border-gray-400"
            >
              Start new session
            </button>
          </div>
        </div>

        {/* Upload Section */}
        {showUploadSection && (
          <div className="bg-gray-50 rounded-lg p-4 mb-3">
            <h3 className="text-lg font-medium text-gray-800 mb-3">Add New Documents</h3>
            
            <div className="mb-4">
              <div 
                className="border-2 border-dashed border-gray-300 rounded-lg p-4 text-center hover:border-gray-400 transition-colors cursor-pointer"
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf"
                  onChange={handleFileSelect}
                  className="hidden"
                />
                <div className="text-gray-600">
                  <svg className="mx-auto h-8 w-8 text-gray-400 mb-2" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <p className="text-sm">Click to select PDF files or drag and drop</p>
                </div>
              </div>
            </div>

            {selectedNewFiles.length > 0 && (
              <div className="mb-4">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Selected Files ({selectedNewFiles.length})
                </h4>
                <div className="space-y-2 max-h-32 overflow-y-auto">
                  {selectedNewFiles.map((file, index) => (
                    <div key={index} className="flex items-center justify-between bg-white p-2 rounded border">
                      <span className="text-sm text-gray-700 truncate">{file.name}</span>
                      <button
                        onClick={() => removeNewFile(index)}
                        className="text-red-500 hover:text-red-700 flex-shrink-0 ml-2"
                      >
                        <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {uploadError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md">
                <p className="text-red-600 text-sm">{uploadError}</p>
              </div>
            )}

            <div className="flex space-x-2">
              <button
                onClick={handleUploadNewFiles}
                disabled={isUploadingFiles || selectedNewFiles.length === 0}
                className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-sm"
              >
                {isUploadingFiles ? 'Uploading...' : `Upload ${selectedNewFiles.length} file(s)`}
              </button>
              <button
                onClick={handleCancelUpload}
                className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
        
        {/* Document List */}
        {documents.length > 0 && (
          <div className="bg-gray-50 rounded-lg p-3">
            <div className="flex items-center justify-between mb-2">
              <button
                onClick={() => setShowDocuments(!showDocuments)}
                className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-gray-900"
              >
                <span>Uploaded Documents</span>
                <svg
                  className={`w-4 h-4 transition-transform ${showDocuments ? 'rotate-90' : ''}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
              <div className="flex items-center space-x-3">
                <span className="text-xs text-gray-500">{documents.length} files</span>
                <span className="text-xs font-medium text-green-600">
                  Cost: ${totalSessionCost.toFixed(4)}
                </span>
              </div>
            </div>
            {showDocuments && (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2">
                {documents.map((doc, index) => (
                <div
                  key={index}
                  className="flex items-center space-x-2 bg-white p-2 rounded border border-gray-200"
                >
                  <div className="flex-shrink-0">
                    <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 100 2h6a1 1 0 100-2H7z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs font-medium text-gray-700 truncate" title={doc.filename}>
                      ID {doc.id}: {doc.filename}
                    </p>
                    <p className="text-xs text-gray-500">
                      {doc.total_pages} pages
                    </p>
                  </div>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDeleteDocument(doc.id);
                    }}
                    className="flex-shrink-0 ml-2 text-red-500 hover:text-red-700 p-1"
                    title="Delete document"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {documents.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center max-w-md">
              <svg className="mx-auto h-16 w-16 text-gray-400 mb-4" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No documents uploaded</h3>
              <p className="text-gray-500 mb-4">
                Upload PDF documents to start chatting about their content. Click "Add Files" above to get started.
              </p>
              <button
                onClick={() => setShowUploadSection(true)}
                className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 4v16m8-8H4" />
                </svg>
                Add Your First Document
              </button>
            </div>
          </div>
        ) : messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <p>Ask me anything about your uploaded documents!</p>
            <p className="text-sm mt-2">Examples:</p>
            <ul className="text-sm mt-1 space-y-1">
              <li>• "What are the main topics covered?"</li>
              <li>• "Summarize the key findings"</li>
              <li>• "What does it say about [specific topic]?"</li>
            </ul>
          </div>
        ) : null}

        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${
              message.role === 'user' ? 'justify-end' : 'justify-start'
            }`}
          >
            <div
              className={`max-w-[80%] p-3 rounded-lg ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-100 text-gray-900'
              }`}
            >
              {/* Progress indicator for streaming messages */}
              {message.isStreaming && message.progress && (
                <div className="mb-3 p-3 bg-blue-50 rounded-lg border-l-4 border-blue-400">
                  <div className="flex items-center space-x-2 mb-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                    <span className="text-sm font-medium text-blue-800">
                      {message.progress.status}
                    </span>
                  </div>
                  
                  <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-500 ease-out"
                      style={{ width: `${(message.progress.step / message.progress.total) * 100}%` }}
                    ></div>
                  </div>
                  
                  <div className="flex justify-between text-xs text-blue-600">
                    <span>Step {message.progress.step} of {message.progress.total}</span>
                    {message.progress.stepTime && message.progress.stepCost && (
                      <span>{message.progress.stepTime.toFixed(2)}s • ${message.progress.stepCost.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              )}

              <div className="prose prose-sm max-w-none">
                <ReactMarkdown
                  components={{
                    p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                    a: ({ href, title, children }) => {
                      // Handle page reference links
                      if (title && title.startsWith('data-page-ref=')) {
                        const pageRef = title.replace('data-page-ref=', '');
                        const [filename, pageSpec] = pageRef.split(':');
                        const pages = parsePageSpecification(pageSpec);
                        
                        if (pages.length > 0) {
                          const pageNum = pages[0]; // Use first page for click
                          return (
                            <button
                              onClick={() => handlePageReference(filename, pageNum)}
                              className="text-blue-600 hover:text-blue-800 underline font-medium cursor-pointer mx-0.5"
                              title={`View ${filename} - Page ${pageNum} content`}
                            >
                              {children}
                            </button>
                          );
                        }
                      }
                      
                      // Regular links
                      return (
                        <a href={href} title={title} className="text-blue-600 hover:text-blue-800 underline">
                          {children}
                        </a>
                      );
                    },
                    h1: ({ children }) => <h1 className="text-lg font-bold mb-2">{children}</h1>,
                    h2: ({ children }) => <h2 className="text-base font-bold mb-2">{children}</h2>,
                    h3: ({ children }) => <h3 className="text-sm font-bold mb-1">{children}</h3>,
                    ul: ({ children }) => <ul className="list-disc ml-4 mb-2">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal ml-4 mb-2">{children}</ol>,
                    li: ({ children }) => <li className="mb-1">{children}</li>,
                    code: ({ children, ...props }) => {
                      const isInline = !props.className?.includes('language-');
                      const bgColor = message.role === 'user' ? 'bg-blue-700 bg-opacity-50' : 'bg-gray-200';
                      return isInline ? (
                        <code className={`${bgColor} px-1 rounded text-xs font-mono`}>{children}</code>
                      ) : (
                        <code className={`block ${bgColor} p-2 rounded text-xs font-mono whitespace-pre-wrap`}>{children}</code>
                      );
                    },
                    strong: ({ children }) => <strong className="font-bold">{children}</strong>,
                    em: ({ children }) => <em className="italic">{children}</em>,
                  }}
                >
                  {processPageReferences(message.content)}
                </ReactMarkdown>
              </div>
              {message.metadata && (
                <div className="mt-2 text-xs opacity-60 space-y-1">
                  <div className="flex flex-wrap gap-x-4 gap-y-1">
                    {message.metadata.selectedDocuments && (
                      <span><span className="font-medium">Docs:</span> {message.metadata.selectedDocuments.map(doc => `${doc.filename} (ID: ${doc.id})`).join(', ')}</span>
                    )}
                    {message.metadata.relevantPagesCount && (
                      <span><span className="font-medium">Pages:</span> {message.metadata.relevantPagesCount}</span>
                    )}
                    {message.metadata.model && (
                      <span><span className="font-medium">Model:</span> {message.metadata.model}</span>
                    )}
                    {message.metadata.timing?.total_time && (
                      <span><span className="font-medium">Time:</span> {message.metadata.timing.total_time.toFixed(1)}s</span>
                    )}
                    {message.metadata.costs?.total_cost && (
                      <span><span className="font-medium">Cost:</span> ${message.metadata.costs.total_cost.toFixed(4)}</span>
                    )}
                  </div>
                </div>
              )}
              <p className={`text-xs mt-1 ${
                message.role === 'user' ? 'text-blue-200' : 'text-gray-500'
              }`}>
                {message.timestamp.toLocaleTimeString()}
              </p>
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-4">
        {/* Model Selection */}
        <div className="mb-3">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            AI Model:
          </label>
          {availableModels.length === 0 ? (
            <p className="text-sm text-gray-500">Loading models...</p>
          ) : (
            <div className="flex flex-wrap gap-3">
              {availableModels.map(model => (
                <label key={model.id} className="flex items-center space-x-2">
                  <input
                    type="radio"
                    name="model"
                    value={model.id}
                    checked={selectedModel === model.id}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    disabled={isLoading}
                    className="text-blue-600 focus:ring-blue-500"
                  />
                  <span className="text-sm text-gray-700">{model.label}</span>
                </label>
              ))}
            </div>
          )}
        </div>
        
        <div className="flex space-x-2">
          <textarea
            value={currentQuestion}
            onChange={(e) => setCurrentQuestion(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={documents.length === 0 ? "Upload documents first to start chatting..." : "Ask a question about your documents..."}
            className="flex-1 px-3 py-2 text-gray-700 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
            rows={2}
            disabled={isLoading || documents.length === 0 || !selectedModel}
          />
          <button
            onClick={handleSendMessage}
            disabled={!currentQuestion.trim() || isLoading || documents.length === 0 || !selectedModel}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </div>
      </div>

      {/* Page Content Modal */}
      {selectedPageContent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-lg max-w-4xl max-h-[80vh] w-full flex flex-col">
            {/* Modal Header */}
            <div className="flex items-center justify-between p-4 border-b border-gray-200">
              <div>
                <h3 className="text-lg font-semibold text-gray-800">
                  {selectedPageContent.filename} - Page {selectedPageContent.pageNumber}
                </h3>
                <p className="text-sm text-gray-600">Extracted Text Content</p>
              </div>
              <button
                onClick={() => setSelectedPageContent(null)}
                className="text-gray-500 hover:text-gray-700 focus:outline-none"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            
            {/* Modal Content */}
            <div className="flex-1 overflow-auto p-6">
              <div className="prose prose-sm max-w-none">
                <div className="whitespace-pre-wrap text-gray-800 leading-relaxed">
                  {selectedPageContent.content}
                </div>
              </div>
            </div>
            
            {/* Modal Footer */}
            <div className="flex justify-end p-4 border-t border-gray-200">
              <button
                onClick={() => setSelectedPageContent(null)}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
} 