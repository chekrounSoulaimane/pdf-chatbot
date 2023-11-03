import { Document } from 'langchain/document';

export type Message = {
  type: 'botMessage' | 'userMessage';
  message: string;
  isStreaming?: boolean;
  sourceDocs?: Document[];
};
