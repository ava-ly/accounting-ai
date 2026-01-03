const API_URL = "http://127.0.0.1:8000";

// Define the structure to match Python
type Message = {
  role: "user" | "ai";
  content: string;
};

export const sendMessageToAgent = async (history: Message[]) => {
  try {
    const response = await fetch(`${API_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ messages: history }),
    });

    if (!response.ok) {
      throw new Error("Network response was not ok");
    }

    const data = await response.json();
    return data.answer; 
  } catch (error) {
    console.error("API Error:", error);
    return "Error: Connection failed.";
  }
};