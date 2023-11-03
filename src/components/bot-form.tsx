import { LoadingDots } from "./loading-dots";
import { useRef, useEffect } from "react";

type Props = {
	handleSubmit: (_e: any) => Promise<void>;
	handleEnter: (_e: any) => void;
	onChange: (_value: string) => void;
	query: string;
	loading: boolean;
};

export function BotForm({
	handleSubmit,
	handleEnter,
	onChange,
	query,
	loading,
}: Props) {
	const textAreaRef = useRef<HTMLTextAreaElement>(null);

	useEffect(() => {
		textAreaRef.current?.focus();
	}, []);

	return (
		<div className="relative flex flex-col justify-center items-center px-4">
			<div className="relative">
				<form onSubmit={handleSubmit}>
					<textarea
						disabled={loading}
						onKeyDown={handleEnter}
						ref={textAreaRef}
						autoFocus={false}
						rows={1}
						maxLength={512}
						id="userInput"
						name="userInput"
						placeholder={
							loading
								? "Waiting for response..."
								: "What is this legal case about?"
						}
						value={query}
						onChange={(e) => onChange(e.target.value)}
						className="w-[75vw] relative resize-none text-base py-4 px-8 bg-white text-black outline-none border border-[#d9d9e3] rounded-lg"
					/>
					<button
						type="submit"
						disabled={loading}
						className="absolute flex top-3.5 right-4 text-[#A5A2A2] p-1"
					>
						{loading ? (
							<div className="absolute top-1 right-1">
								<LoadingDots color="#000" />
							</div>
						) : (
							// Send icon SVG in input field
							<svg
								viewBox="0 0 20 20"
								className="w-[1.2em] h-[1.2em] rotate-90 fill-current"
								xmlns="http://www.w3.org/2000/svg"
							>
								<path d="M10.894 2.553a1 1 0 00-1.788 0l-7 14a1 1 0 001.169 1.409l5-1.429A1 1 0 009 15.571V11a1 1 0 112 0v4.571a1 1 0 00.725.962l5 1.428a1 1 0 001.17-1.408l-7-14z"></path>
							</svg>
						)}
					</button>
				</form>
			</div>
		</div>
	);
}
