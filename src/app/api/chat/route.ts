import { ChatOpenAI } from "langchain/chat_models/openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { AIMessage, HumanMessage } from "langchain/schema";
import { pinecone } from "@/libs/pinecone";

const CONDENSE_TEMPLATE = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_TEMPLATE = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;

export async function POST(request: Request) {
	const { question, history } = await request.json();
	// const { question, history } = request.body;

	console.log("question", question);
	console.log("history", history);

	if (!question) {
		return Response.json(
			{ message: "No question in the request" },
			{ status: 200 }
		);
	}

	return Response.json({ error: "Nothing here" }, { status: 200 });
}
