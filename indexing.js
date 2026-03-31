import * as dotenv from 'dotenv';
dotenv.config();
import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import { HuggingFaceTransformersEmbeddings } from "@langchain/community/embeddings/hf_transformers";


async function indexing() {
    
    // pdf file ko load kariye
    const PDF_PATH = './Node.pdf';
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();

    
    //  chunking create karna
    

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 300,
        chunkOverlap: 50,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);

    // console.log(chunkedDocs.length); 266 chunk --> vector

    // embeding create karni hai
    // configure kar diya hai

    const embeddings = new HuggingFaceTransformersEmbeddings({
        model: "Xenova/all-MiniLM-L6-v2",});

   

    // configure pinecone

    const pinecone = new Pinecone();
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);


    // single step--> ChunkedDocs-->Embedding --> Vector DB

    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });
}

indexing();