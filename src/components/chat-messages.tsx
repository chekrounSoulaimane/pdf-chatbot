import Image from "next/image";
import ReactMarkdown from "react-markdown";
import { RefObject } from "react";
import { Document } from "langchain/document";
import {
	Accordion,
	AccordionContent,
	AccordionItem,
	AccordionTrigger,
} from "@/components/accordion";
import { Message } from "@/types/message.type";

type Props = {
	messages: Message[];
	loading: boolean;
	messageListRef: RefObject<HTMLDivElement>;
};

export function ChatMessages({ messages, loading, messageListRef }: Props) {
	return (
		<div className="grow relative w-full">
			<div
				ref={messageListRef}
				className="w-full h-full overflow-y-scroll rounded-lg"
			>
				{messages.map((message, index) => {
					let icon;
					let className = "flex gap-2 items-center ";
					if (message.type === "botMessage") {
						icon = (
							<Image
								key={index}
								src="/bot-image.png"
								alt="AI"
								width="40"
								height="40"
								priority
							/>
						);
						className += "bg-[#f9fafb] text-black p-6 animate-[fadein 0.5s]";
					} else {
						icon = (
							<Image
								key={index}
								src="/usericon.png"
								alt="Me"
								width="30"
								height="30"
								className="h-full mr-4 rounded-sm"
								priority
							/>
						);
						// The latest message sent by the user will be animated while waiting for a response
						className +=
							loading && index === messages.length - 1
								? "usermessagewaiting"
								: "bg-white text-black p-6";
					}
					return (
						<div key={`chatMessage-${index}`}>
							<div className={className}>
								{icon}
								<div className="leading-4">
									<ReactMarkdown>{message.message}</ReactMarkdown>
								</div>
							</div>
							{message.sourceDocs && (
								<div className="p-5" key={`sourceDocsAccordion-${index}`}>
									<Accordion type="single" collapsible className="flex-col">
										{message.sourceDocs.map((doc: Document, index: number) => (
											<div key={`messageSourceDocs-${index}`}>
												<AccordionItem value={`item-${index}`}>
													<AccordionTrigger>
														<h3>Source {index + 1}</h3>
													</AccordionTrigger>
													<AccordionContent>
														<ReactMarkdown>{doc.pageContent}</ReactMarkdown>
														<p className="mt-2">
															<b>Source:</b> {doc.metadata.source}
														</p>
													</AccordionContent>
												</AccordionItem>
											</div>
										))}
									</Accordion>
								</div>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}
