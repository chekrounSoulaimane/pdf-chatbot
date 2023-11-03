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
	// OpenAI recommends replacing newlines with spaces for best results
	const sanitizedQuestion = question.trim().replaceAll("\n", " ");

	try {
		if (!process.env.PINECONE_INDEX_NAME) {
			throw new Error("Missing Pinecone index name in .env file");
		}
		const index = pinecone.Index(process.env.PINECONE_INDEX_NAME);

		/* create vectorstore*/
		const vectorstore = await PineconeStore.fromExistingIndex(
			new OpenAIEmbeddings({}),
			{
				pineconeIndex: index,
				textKey: "text",
			}
		);

		//create chain
		const model = new ChatOpenAI({
			temperature: 0, // increase temperature to get more creative answers
			modelName: "gpt-3.5-turbo", //change this to gpt-4 if you have access
		});

		const chain = ConversationalRetrievalQAChain.fromLLM(
			model,
			vectorstore.asRetriever(),
			{
				qaTemplate: QA_TEMPLATE,
				questionGeneratorTemplate: CONDENSE_TEMPLATE,
				returnSourceDocuments: true, //The number of source documents returned is 4 by default
			}
		);

		const pastMessages = history.map((message: string, i: number) => {
			if (i % 2 === 0) {
				return new HumanMessage(message);
			} else {
				return new AIMessage(message);
			}
		});

		//Ask a question using chat history
		const response = await chain.call({
			question: sanitizedQuestion,
			chat_history: pastMessages,
		});

		console.log("response", response);
		return Response.json(response, { status: 200 });
	} catch (error: any) {
		console.log("error", error);
		return Response.json(
			{ error: error.message || "Something went wrong" },
			{ status: 500 }
		);
	}
}
