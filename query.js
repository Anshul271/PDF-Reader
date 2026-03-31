import readlineSync, { question } from 'readline-sync';
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';
dotenv.config();
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnableSequence } from '@langchain/core/runnables';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";
import { GoogleGenAI, Type } from "@google/genai";

const ai = new GoogleGenAI({
  apiKey: process.env.GEMINI_API_KEY,
});

// 🔥 Memory
let chatHistory = [];

// 🔥 Embeddings
const embeddings = new HuggingFaceTransformersEmbeddings({
  model: "Xenova/all-MiniLM-L6-v2",
});

// 🔥 LLM for RAG
const model = new ChatGoogleGenerativeAI({
  apiKey: process.env.GEMINI_API_KEY,
  model: "gemini-2.5-flash",
  temperature: 0.3,
});

// 🔥 Pinecone
const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);



// ================= INTENT / REWRITE =================
async function rewriteQuery(question) {
  const historyText = chatHistory.join("\n");

  const result = await ai.models.generateContent({
    model: "gemini-2.5-flash",
    contents: [
      {
        role: "user",
        parts: [
          {
            text: `
You are an intent understanding system.

Your job:
- Use chat history
- Understand user question
- Resolve "it", "this", etc.
- Convert into standalone query

IMPORTANT:
- DO NOT answer
- ONLY rewrite

Chat History:
${historyText}

User Question:
${question}

Rewritten Question:
            `,
          },
        ],
      },
    ],
  });

  return result.text.trim();
}



// ================= RAG =================
async function chatting(question) {

  // intent model ko introduce: Homework

  // question ki embedding create karna hai
  const queryVector = await embeddings.embedQuery(question);

  // embeddig aagyi, uske baad usko vectorDB ke andar search karna, top10
  const searchResults = await pineconeIndex.query({
    topK: 10,
    vector: queryVector,
    includeMetadata: true,
  });

  const context = searchResults.matches
    .map(match => match.metadata.text)
    .join("\n\n---\n\n");

  // console.log(searchResults);

  // top10+question isko mein llm ko de dunga

  const promptTemplate = PromptTemplate.fromTemplate(`
You are a helpful assistant answering questions based on the provided documentation.

Context from the documentation:
{context}

Question: {question}

Instructions:
- Answer the question using ONLY the information from the context above
- If the answer is not in the context, say "I don't have enough information to answer that question."
- Be concise and clear
- Use code examples from the context if relevant

Answer:
  `);

  const chain = RunnableSequence.from([
    promptTemplate,
    model,
    new StringOutputParser(),
  ]);

  // Step 6: Invoke the chain and get the answer
  const answer = await chain.invoke({
    context: context,
    question: question,
  });

  console.log(answer);

  // Output create kar dunga
}



// ================= MAIN =================
async function main() {
  const userInput = readlineSync.question("Ask me anything--> ");

  // 🔥 Step 1: Rewrite query
  const rewritten = await rewriteQuery(userInput);
  console.log("\n🔁 Rewritten:", rewritten);

  // 🔥 Step 2: RAG
  await chatting(rewritten);

  // 🔥 Step 3: Save memory
  chatHistory.push(`User: ${userInput}`);
  chatHistory.push(`Rewritten: ${rewritten}`);

  main();
}

main();