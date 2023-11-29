# LangChain

 - modules (python) / components (javascript)

### 1 - LLM Components

 - model io - focuses on basic LLM I/O (support text completion and chat interface)
 - data connection - connects LLMs to a data source (e.g. vector store / docs)
 - chains - link the output of one model to be the input of another model
 - memory - allows models to retain historical context of previous interactions
 - agents - LLM-based apps that perform complex tasks - group Models, Data Connections, Chains and Memory to reason through requests and perform actions on observational outputs
 - *Prompt Flow*
   - System > Human > Compile ChatTemplate > Insert Topic > Run

### 2 - Data Connections

 - Document loading and Integrations - built-in are for the common formats, plus integrations for various data sources
 - Document Transformers - length of the strings may be too large to feed into a model, both embeddings and chat models - transformers allow to split documents into chunks that can be served as useful components for embeddings
 - Text Embedding - convert string text into an embedded vectorized representation that can be stored and then used for similarity matches against them 
   - different embedding models can't interact with e.o., meaning you would need to re-embed the entire set of documents if you were to switch the models
 - Vector Stores - 
   - Can store key attributes - support N-dim vectors, can index an embedded vector to its associated text doc, can 'query' (cosine similarity search b/w a new vector not in the DB), supports CRUD for new vectors   
   - Vector stores can be passed as "retrievers" via `as_retriever()`
   - *Vector Flow*
     - Load Document > Split into Chunks 
     - Create Embedding > Embed Chunks > Vectors
     - Vector Chunks > Save to DB
     - "query" > similarity search
 - Queries and Retrievers
   - compression - using LLM to distill a vector DB response into a more relevant output
   - *Compression Flow*
     - LLM Use Compression
     - LLM > LLMChainExtractor
     - Contextual Compression

### 3 - Chains
 - Chains allow linking of the outputs of one LLM call as the inputs for another call
 - LLMChain / SimpleSequentialChain / SequentialChain - basic building block
 - LLMRouterChain - take an input and redirect to the most appropriate LLMChain sequence
 - TransformChain - allows to insert custom functionality 
 - create_structured_output_chain - allows calling OpenAI functions from the models (e.g. return json directly form the response)
 - MathChain - for math questions - `LLMathChain.from_llm`
 - QAChain - allows running Q&A on a vector store  

### 4 - Memory
 - Keeps track of the previous conversations - via storing all messages, some interactions or tokens
 - `langchain.memory.ChatMessageHistory` - `add_ai_message`, `add_user_message`, etc.

### 5 - Agents
 - ReACT framework - LLMs to connect to tools and construct a structured approach to complete a task based on reasoning and acting
 - Multiple types: conversational, P&E, ReAct, etc.
 - Conversational agents - keep history of the conversation