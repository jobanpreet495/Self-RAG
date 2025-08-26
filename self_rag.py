from openai import OpenAI 
from typing  import List , Tuple , Dict , Any
from standard_retrieval import StandardRetriever
from models import  ShouldRetrieve , RelevanceAssessment,SupportAssessment,UtilityAssessment,RAGAnswer
from dotenv import load_dotenv 
load_dotenv()



class SelfRAGModel:
    """SELF-RAG Implementation following the research paper"""
    
    def __init__(self,urls: List[str], model_name: str = "gpt-4o"):
        self.model_name = model_name
        self.retriever = StandardRetriever(urls)
        # self.reflection_tokens = ReflectionTokens()
        self.client = OpenAI() 
        
    def call_llm(self, prompt: str,pydantic_class) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.responses.parse(
                model=self.model_name,
                input=[{"role": "user", "content": prompt}],
                text_format = pydantic_class,
                temperature = 0
               
            )
            return response.output_parsed
        except Exception as e:
            print(f"LLM API error: {e}")
            return "Error: Could not generate response"
    
    def should_retrieve(self, query: str) -> Tuple[bool, str]:
        """Determine if retrieval is needed (Step 1 in paper)"""
        prompt = f"""You are an expert assistant tasked with determining whether external document retrieval is necessary to answer a user query effectively.

            **Instructions:**
            Analyze the given query to decide if retrieving external documents would significantly improve the response quality.
            
            **Decision Criteria:**
            - Choose "Yes" if the query requires:
              * Specific factual information not in general knowledge
              * Recent events, news, or time-sensitive data
              * Technical specifications, statistics, or domain-specific details
              * Citations from specific sources or documents
              * Information that may have changed since your training data
            
            - Choose "No" if the query can be answered using:
              * General knowledge and common facts
              * Logical reasoning and analysis
              * Creative tasks (writing, brainstorming)
              * Explanations of well-established concepts
              * Personal advice or opinions
            
            **Query:** {query}
            
            
            Please provide your decision, a clear explanation of your reasoning."""
        
        response = self.call_llm(prompt,ShouldRetrieve)
        return response.decision
    
    def assess_relevance(self, query: str, document: Dict[str, str]) -> RelevanceAssessment:
        """Assess if retrieved document is relevant (ISREL token)"""
        
        prompt = f"""You are an expert document relevance assessor. Your task is to determine if the provided evidence document is relevant and useful for answering the given instruction.

            **Assessment Guidelines:**
            - Choose "Relevant" if the document:
              * Contains direct answers or information that addresses the instruction
              * Provides supporting facts, data, or context needed to complete the task
              * Includes relevant background information that enhances understanding
              * Contains specific details, examples, or evidence related to the query
            
            - Choose "Irrelevant" if the document:
              * Contains information unrelated to the instruction
              * Only mentions the topic tangentially without useful details
              * Provides information that doesn't help answer the question
              * Is primarily about different topics or concepts
            
            **Instruction:** {query}
            
            **Evidence Document:**
            {document.get('page_content')}
            
            Based on the instruction and evidence provided, determine the relevance of this document."""

        response = self.call_llm(prompt, RelevanceAssessment)
        return response.relevance
    
    def assess_support(self, query: str, generation: str, document: Dict[str, str]) -> SupportAssessment:
        """Assess if generation is supported by document (ISSUP token)"""
        
        prompt = f"""You are an expert fact-checker tasked with evaluating whether a generated response is supported by the provided evidence document.

                **Evaluation Guidelines:**
                
                - Choose "Fully supported" if:
                  * Every factual claim in the response can be verified from the evidence
                  * All key information is directly backed by the document content
                  * No unsupported assertions or additions beyond the evidence
                  * The response stays within the bounds of what the evidence provides
                
                - Choose "Partially supported" if:
                  * Some claims in the response are supported by the evidence
                  * Major portions of the response lack evidence backing
                  * The response contains a mix of supported and unsupported information
                  * Core facts are correct but additional details are unverified
                
                - Choose "No support" if:
                  * The response contradicts information in the evidence
                  * Claims are completely unrelated to the evidence content
                  * The response ignores the evidence entirely
                  * Factual information directly conflicts with the document
                
                **Instruction:** {query}
                
                **Generated Response:**
                {generation}
                
                **Evidence Document:**
                {document.get('page_content')}
                
                Evaluate how well the generated response is supported by the evidence provided."""

        response = self.call_llm(prompt, SupportAssessment) 
        return response.support_level
        
    
    def assess_utility(self, query: str, generation: str) -> UtilityAssessment:
        """Assess utility of generation (ISUSE token)"""
        
        prompt = f"""You are an expert response evaluator tasked with rating the utility and helpfulness of a generated response to a given instruction.

            **Evaluation Scale (1-5):**
            
            **5 - Excellent Utility:**
            - Completely addresses all aspects of the instruction
            - Highly detailed, comprehensive, and informative
            - Provides actionable insights or thorough explanations
            - Goes above and beyond to be maximally helpful
            - Clear, well-structured, and easy to understand
            
            **4 - Good Utility:**
            - Addresses most aspects of the instruction effectively  
            - Generally detailed and informative
            - Minor improvements or additions could enhance completeness
            - Helpful and relevant throughout
            - Well-organized response
            
            **3 - Acceptable Utility:**
            - Addresses the main request adequately
            - Provides basic information but lacks depth
            - Major additions needed for completeness
            - Generally relevant but could be more comprehensive
            - Meets minimum expectations
            
            **2 - Limited Utility:**
            - Addresses main request but incomplete or superficial
            - Missing key information or context
            - Not fully relevant to all aspects of the instruction
            - Requires significant improvements to be truly helpful
            - Partially useful but inadequate
            
            **1 - Poor Utility:**
            - Barely addresses the instruction or completely off-topic
            - Lacks relevant information or is misleading
            - Unhelpful or confusing response
            - Does not meet the user's needs
            - Irrelevant or inappropriate content
            
            **Instruction:** {query}
            
            **Generated Response:**
            {generation}
            
            Rate the utility of this response based on how well it addresses the instruction and provides helpful, relevant information."""

        response = self.call_llm(prompt, UtilityAssessment)
        
        return response.utility_score
        
    def generate_with_context(self, query: str, documents: List[Dict[str, str]] = None) -> str:
        """Generate response with optional retrieved context.
        - If documents are provided → use ONLY the context.
        - If no documents are provided → answer based on model's parametric knowledge.
        """
        
        if documents and len(documents) > 0:
            # Build rich context with document metadata
            context_sections = []
            for doc in documents:
                content = doc.get('page_content')                
                context_sections.append(content)
            
            context = "\n\n".join(context_sections)
            
            prompt = f"""You are a helpful assistant. Use ONLY the provided context to answer the question. 
                    If the answer is not in the context, say: "The information is not available in the provided context."
                    
                    Context:
                    {context}
                    
                    Question:
                    {query}
                    
                    Answer in a clear, concise, and factual way:
                    
                    **Answer:**"""
        else:
            # No documents provided → rely on model's parametric knowledge
            prompt = f"""You are a helpful assistant. Answer the following question using your own knowledge. 
                        If the question is unclear or you don’t know, say: "I don't know the answer to that."
                        
                        Question:
                        {query}
                        
                        Answer in a clear, concise, and factual way:
                        
                        **Answer:**"""
    
        response = self.call_llm(prompt, RAGAnswer)  
        return response.answer

        

    
    def generate(self, query: str) -> Dict[str, Any]:
        """
        Main SELF-RAG generation pipeline following Algorithm 1 from paper
        """
        print(f"🔍 Processing query: {query}")
        print("="*60)
        
        # Step 1: Decide if retrieval is needed
        should_retrieve_response = self.should_retrieve(query)
        print(f"🤔 Should retrieve: {should_retrieve_response}")
        
        if should_retrieve_response.lower() == 'yes':
            # Step 2: Retrieve documents
            print("📚 Retrieving documents...")
            documents = self.retriever.retrieve(query)
            print(f"📄 Retrieved {len(documents)} documents")
            
            if not documents:
                # No documents found, fallback to generation without retrieval
                print("⚠️ No documents retrieved, falling back to generation without context")
                generation = self.generate_with_context(query, None)
                utility_score = self.assess_utility(query, generation)
                
                result = {
                    "final_answer": generation,
                    "used_retrieval": False,
                }
                return result
            
            # Step 3: Process each document and find the best generation
            best_generation = ""
            best_score = -1
            # best_tokens = []
            all_generations = []
            
            for doc_idx, document in enumerate(documents):
                print(f"  📄 Processing document {doc_idx + 1}")
                
                # Assess relevance
                relevance_assessment = self.assess_relevance(query, document.dict())
                print(f"    📊 Relevance: {relevance_assessment}")
                
                if relevance_assessment.lower() == "relevant":
                    # Generate response using this document
                    generation_response = self.generate_with_context(query, [document.dict()])
                    
                    # Assess support
                    support_assessment = self.assess_support(query, generation_response, document.dict())
                    print(f"    🔍 Support: {support_assessment}")
                    
                    # Assess utility
                    utility_score = self.assess_utility(query, generation_response)
                    print(f"    ⭐ Utility: {utility_score}")
                    
                    # Calculate score based on support and utility
                    score = 0
                    if support_assessment.lower() == "fully supported":
                        score += 3
                    elif support_assessment.lower() == "partially supported":
                        score += 1
                    
                    # Extract numeric utility score
                    if isinstance(utility_score, str):
                        if "5" in utility_score:
                            score += 3
                        elif "4" in utility_score:
                            score += 2
                        elif "3" in utility_score:
                            score += 1
                    else:
                        # If utility_score is already numeric
                        score += max(0, int(utility_score) - 2)
                    
                    # Store this generation
                    generation_info = {
                        "text": generation_response,
                        "score": score,
                        "doc_index": doc_idx
                    }
                    all_generations.append(generation_info)
                    
                    # Update best generation if this one is better
                    if score > best_score:
                        best_score = score
                        best_generation = generation_response
                        print(f"    ✨ New best generation (score: {best_score})")
                else:
                    print(f"    ❌ Document marked as irrelevant, skipping generation")
            
            # Return the best generation if we found any relevant documents
            if best_generation:
                result = {
                    "final_answer": best_generation,
                    "used_retrieval": True,
                    "total_score": best_score,
                    # "all_generations": all_generations
                }
                print(f"🎯 Final result with retrieval (score: {best_score})")
                return result
            else:
                # No relevant documents found, fallback to generation without retrieval
                print("⚠️ No relevant documents found, falling back to generation without context")
                generation_response = self.generate_with_context(query, None)
                utility_score = self.assess_utility(query, generation_response)
                
                result = {
                    "final_answer": generation_response,
                    "used_retrieval": False,
                }
                return result
        
        else:
            # Step 4: Generate without retrieval
            print("🚫 Retrieval not needed, generating directly")
            generation_response = self.generate_with_context(query)
            print(generation_response)
            utility_score = self.assess_utility(query, generation_response)
            print(f"✨ Utility: {utility_score}")
            
            result = {
                "final_answer": generation_response,
                "used_retrieval": False,
            }
            print("🎯 Final result without retrieval")
            return result