import os
import openai
import tiktoken
 
openai.api_key = os.getenv("OPENAI_KEY")

class ChatBot:
 
    def __init__(self, message):
        self.messages = []
        self.system_message = message

    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content
        })

    def remove_last_message(self):
        self.messages = self.messages[:-1]

    def remove_messages(self, role):
        self.messages = [message for message in self.messages if message["role"] != role]

    def askForInput(self):
        tokens = self.num_tokens_from_messages(self.messages)
        # print(f"Total tokens: {tokens}\n")
 
        if tokens > 4000:
            # print("WARNING: Number of tokens exceeds 4000. Truncating messages.")
            self.messages = self.messages[2:]

        self.remove_messages("system")
        prompt = input("You: ")

        if prompt == "regenerate":
            print("\nRegenerating...\n")
            self.remove_last_message()
        else:
            print("\nGenerating...\n")
            self.add_message("user", prompt)

        self.add_message("system", self.system_message)
         
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages = self.messages,
            temperature = 1,
            frequency_penalty = 1,
        )
         
        answer = response.choices[0]['message']['content']
         
        self.add_message("assistant", answer)

    def printMessagesToConsole(self):
        for message in self.messages:
            print(f"{message['role'].capitalize()}: {message['content']}\n")

    def num_tokens_from_messages(self, messages, model="gpt-3.5-turbo"):
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":  # note: future models may deviate from this
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {model}.""")


if __name__ == "__main__":
    bot = ChatBot(input("System Prompt: "))

    while True:
        bot.askForInput()
        bot.printMessagesToConsole()