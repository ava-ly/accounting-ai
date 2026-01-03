import { useState, useRef, useEffect } from "react";
import { Send, Bot, User, Loader2, Sparkles } from "lucide-react";
import ReactMarkdown from 'react-markdown';
import { sendMessageToAgent } from "./api";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Card } from "@/components/ui/card";

type Message = {
  role: "user" | "ai";
  content: string;
};

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "ai",
      content: "Xin chào! I am **Accounting AI**. \n\nI can help you with:\n- **VAS/IFRS** lookups\n- **Tax Calculations**\n- **Journal Entries**\n\nHow can I help you today?"
    }
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsgContent = input;
    setInput("");
    setIsLoading(true);

    const newUserMsg = { role: "user" as const, content: userMsgContent };

    const newHistory = [...messages, newUserMsg];
    setMessages(newHistory)

    const aiResponse = await sendMessageToAgent(newHistory);
    setMessages((prev) => [ ...prev, { role: "ai", content: aiResponse }]);
    setIsLoading(false);
  }

  return (
    <div className="flex h-screen bg-slate-50 items-center justify-center p-4">
      {/* Main Chat Container */}
      <Card className="w-full max-w-4xl h-[90vh] flex flex-col shadow-xl overflow-hidden border-slate-200">
        {/* --- HEADER --- */}
        <div className="p-4 bg-white flex items-center justify-between border-b">
          <div className="flex items-center gap-3">
            <div className="bg-primary/10 p-2 rounded-lg">
              <Bot className="text-primary h-6 w-6" />
            </div>
            <div>
              <h1 className="font-bold text-lg text-slate-800">Accounting Assistant</h1>
              <div className="flex items-center gap-1.5">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                </span>
                <span className="text-xs text-slate-500 font-medium">Meta Llama 4 Scout 17B</span>
              </div>
            </div>
          </div>
          <Button variant="outline" size="icon" title="Clear Chat" onClick={() => window.location.reload()}>
            <Sparkles className="h-4 w-4 text-slate-500" />
          </Button>
        </div>

        {/* --- CHAT AREA --- */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 bg-slate-50/50">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex gap-3 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}
            >
              {/* Avatar */}
              <Avatar className="h-9 w-9 border">
                {msg.role === "ai" ? (
                  <>
                    <AvatarImage src="/bot-icon.png" />
                    <AvatarFallback className="bg-primary/10 text-primary"><Bot size={18} /></AvatarFallback>
                  </>
                ) : (
                  <AvatarFallback className="bg-slate-200 text-slate-600"><User size={18} /></AvatarFallback>
                )}
              </Avatar>

              {/* Message Bubble */}
              <div
                className={`max-w-[85%] px-4 py-3 rounded-2xl text-[15px] shadow-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-primary text-primary-foreground rounded-tr-sm"
                    : "bg-white border border-slate-200 text-slate-800 rounded-tl-sm"
                }`}
              >
                {msg.role === "ai" ? (
                  <div className="prose prose-sm max-w-none prose-slate dark:prose-invert">
                    {/* SAFETY CHECK: Only render if content is a string */}
                    {msg.content && typeof msg.content === 'string' ? (
                      // Try adding the plugin back later, but keep it simple for now
                      <ReactMarkdown>
                        {msg.content}
                      </ReactMarkdown>
                    ) : (
                      <span className="text-red-400 italic">
                        Error: Could not render response. (Raw: {JSON.stringify(msg.content)})
                      </span>
                    )}
                  </div>
                ) : (
                  msg.content
                )}
              </div>
            </div>
          ))}

          {/* Loading Indicator */}
          {isLoading && (
            <div className="flex gap-3">
              <Avatar className="h-9 w-9">
                <AvatarFallback className="bg-primary/10"><Bot size={18} /></AvatarFallback>
              </Avatar>
              <div className="bg-white border border-slate-200 px-4 py-3 rounded-2xl rounded-tl-sm shadow-sm flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
                <span className="text-sm text-slate-500 font-medium">Analyzing legislation...</span>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* --- INPUT AREA --- */}
        <div className="p-4 bg-white border-t">
          <div className="flex gap-2">
            <Input 
              placeholder="Ask about Account 111, VAT calculation, or circulars..." 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              className="flex-1 bg-slate-50 border-slate-200 focus-visible:ring-primary"
              disabled={isLoading}
            />
            <Button 
              onClick={handleSend} 
              disabled={isLoading || !input.trim()}
              className="bg-primary hover:bg-primary/90"
            >
              <Send className="h-4 w-4 mr-2" />
              Send
            </Button>
          </div>
          <div className="mt-2 text-center">
             <p className="text-[10px] text-slate-400">
               AI can make mistakes. Please verify with official documents.
             </p>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default App;
