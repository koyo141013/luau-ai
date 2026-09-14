const chatArea = document.getElementById("chatArea");
const promptInput = document.getElementById("prompt");
const sendButton = document.getElementById("sendButton");
const welcome = document.getElementById("welcome");

const modeName = document.getElementById("modeName");
const modeIcon = document.getElementById("modeIcon");

const newChat = document.getElementById("newChat");

const toast = document.getElementById("toast");

let currentMode = "CHAT";

const modeInfo = {
    CHAT: {
        name: "Chat",
        icon: "💬"
    },

    CODE: {
        name: "Code",
        icon: "💻"
    },

    EXPLAIN: {
        name: "Explain",
        icon: "📖"
    },

    FIX: {
        name: "Fix",
        icon: "🔧"
    }
};


/* =========================
   MODE
========================= */

document.querySelectorAll(".mode").forEach(button => {

    button.addEventListener("click", () => {

        document
            .querySelectorAll(".mode")
            .forEach(x => x.classList.remove("active"));

        button.classList.add("active");

        currentMode = button.dataset.mode;

        modeName.textContent =
            modeInfo[currentMode].name;

        modeIcon.textContent =
            modeInfo[currentMode].icon;

        updatePlaceholder();
    });

});


function updatePlaceholder() {

    const placeholders = {

        CHAT:
            "AI에게 메시지를 입력하세요...",

        CODE:
            "만들고 싶은 Luau 코드를 설명하세요...",

        EXPLAIN:
            "설명받고 싶은 Luau 코드를 입력하세요...",

        FIX:
            "고치고 싶은 Luau 코드를 입력하세요..."
    };

    promptInput.placeholder =
        placeholders[currentMode];
}


/* =========================
   SEND
========================= */

sendButton.addEventListener(
    "click",
    sendMessage
);


promptInput.addEventListener(
    "keydown",
    event => {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendMessage();
        }

    }
);


/* =========================
   AUTO RESIZE
========================= */

promptInput.addEventListener(
    "input",
    () => {

        promptInput.style.height = "auto";

        promptInput.style.height =
            Math.min(
                promptInput.scrollHeight,
                160
            ) + "px";

    }
);


/* =========================
   SEND MESSAGE
========================= */

async function sendMessage() {

    const text =
        promptInput.value.trim();

    if (!text) {
        return;
    }

    removeWelcome();

    addUserMessage(text);

    promptInput.value = "";

    promptInput.style.height = "auto";

    const loading = addLoading();

    sendButton.disabled = true;

    try {

        /*
         * Python 서버와 연결
         */

        const response = await fetch(
            "/api/generate",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    mode: currentMode,

                    prompt: text

                })
            }
        );


        if (!response.ok) {

            throw new Error(
                "서버 오류: " +
                response.status
            );
        }


        const data =
            await response.json();


        loading.remove();


        addAIMessage(
            data.response || "응답이 없습니다."
        );


    } catch (error) {

        loading.remove();

        addAIMessage(
            "⚠️ AI 서버에 연결할 수 없습니다.\n\n" +
            "Python 서버가 실행되고 있는지 확인해주세요."
        );

        console.error(error);

    }


    sendButton.disabled = false;

    promptInput.focus();

}


/* =========================
   USER MESSAGE
========================= */

function addUserMessage(text) {

    const message =
        document.createElement("div");

    message.className =
        "message user";


    const bubble =
        document.createElement("div");

    bubble.className =
        "bubble";

    bubble.textContent =
        text;


    message.appendChild(bubble);

    chatArea.appendChild(message);

    scrollBottom();
}


/* =========================
   AI MESSAGE
========================= */

function addAIMessage(text) {

    const message =
        document.createElement("div");

    message.className =
        "message ai";


    const avatar =
        document.createElement("div");

    avatar.className =
        "ai-avatar";

    avatar.textContent =
        "🤖";


    const content =
        document.createElement("div");

    content.className =
        "ai-content";


    const label =
        document.createElement("div");

    label.className =
        "ai-label";

    label.textContent =
        "LUAU AI";


    content.appendChild(label);


    /*
     * CODE 모드에서는
     * 코드 블록으로 표시
     */

    if (
        currentMode === "CODE" ||
        currentMode === "FIX"
    ) {

        const code =
            createCodeBlock(text);

        content.appendChild(code);

    } else {

        const textElement =
            document.createElement("div");

        textElement.className =
            "ai-text";

        textElement.textContent =
            text;

        content.appendChild(textElement);
    }


    message.appendChild(avatar);

    message.appendChild(content);

    chatArea.appendChild(message);

    scrollBottom();
}


/* =========================
   CODE BLOCK
========================= */

function createCodeBlock(text) {

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "code-block";


    const header =
        document.createElement("div");

    header.className =
        "code-header";


    const language =
        document.createElement("span");

    language.textContent =
        "LUAU";


    const copy =
        document.createElement("button");

    copy.className =
        "copy-button";

    copy.textContent =
        "복사";


    const pre =
        document.createElement("pre");


    const code =
        document.createElement("code");

    code.textContent =
        cleanCode(text);


    copy.addEventListener(
        "click",
        async () => {

            await navigator.clipboard.writeText(
                cleanCode(text)
            );

            showToast(
                "코드가 복사되었습니다!"
            );

        }
    );


    header.appendChild(language);

    header.appendChild(copy);

    pre.appendChild(code);

    wrapper.appendChild(header);

    wrapper.appendChild(pre);


    return wrapper;
}


/* =========================
   CLEAN CODE
========================= */

function cleanCode(text) {

    return text
        .replace(/```lua/gi, "")
        .replace(/```luau/gi, "")
        .replace(/```/g, "")
        .trim();

}


/* =========================
   LOADING
========================= */

function addLoading() {

    const message =
        document.createElement("div");

    message.className =
        "message ai";


    const avatar =
        document.createElement("div");

    avatar.className =
        "ai-avatar";

    avatar.textContent =
        "🤖";


    const content =
        document.createElement("div");

    content.className =
        "ai-content";


    const label =
        document.createElement("div");

    label.className =
        "ai-label";

    label.textContent =
        "LUAU AI";


    const typing =
        document.createElement("div");

    typing.className =
        "typing";


    for (let i = 0; i < 3; i++) {

        const dot =
            document.createElement("span");

        typing.appendChild(dot);
    }


    content.appendChild(label);

    content.appendChild(typing);

    message.appendChild(avatar);

    message.appendChild(content);

    chatArea.appendChild(message);

    scrollBottom();


    return message;
}


/* =========================
   WELCOME
========================= */

function removeWelcome() {

    if (welcome) {

        welcome.remove();

    }

}


/* =========================
   QUICK ACTION
========================= */

document
    .querySelectorAll(".quick")
    .forEach(button => {

        button.addEventListener(
            "click",
            () => {

                promptInput.value =
                    button.dataset.prompt;

                promptInput.focus();

                promptInput.dispatchEvent(
                    new Event("input")
                );

            }
        );

    });


/* =========================
   NEW CHAT
========================= */

newChat.addEventListener(
    "click",
    () => {

        chatArea.innerHTML = "";

        chatArea.appendChild(
            createWelcome()
        );

        promptInput.value = "";

        promptInput.focus();

    }
);


function createWelcome() {

    const div =
        document.createElement("div");

    div.className =
        "welcome";

    div.innerHTML = `

        <div class="welcome-icon">
            🤖
        </div>

        <h1>
            Luau AI
        </h1>

        <p>
            Roblox와 Luau 코딩을 도와주는<br>
            나만의 로컬 AI
        </p>

        <div class="quick-actions">

            <button class="quick"
                data-prompt="플레이어가 게임에 들어오면 이름을 출력해줘.">
                <span>💻</span>
                플레이어 이름 출력
            </button>

            <button class="quick"
                data-prompt="local Players = game:GetService('Players')">
                <span>📖</span>
                이 코드 설명해줘
            </button>

            <button class="quick"
                data-prompt="안녕?">
                <span>💬</span>
                AI와 대화하기
            </button>

        </div>
    `;


    div
        .querySelectorAll(".quick")
        .forEach(button => {

            button.addEventListener(
                "click",
                () => {

                    promptInput.value =
                        button.dataset.prompt;

                    promptInput.focus();

                }
            );

        });


    return div;
}


/* =========================
   SCROLL
========================= */

function scrollBottom() {

    setTimeout(() => {

        chatArea.scrollTo({

            top:
                chatArea.scrollHeight,

            behavior:
                "smooth"

        });

    }, 50);

}


/* =========================
   TOAST
========================= */

function showToast(text) {

    toast.textContent =
        text;

    toast.classList.add("show");


    setTimeout(() => {

        toast.classList.remove("show");

    }, 1800);

}


/* =========================
   START
========================= */

updatePlaceholder();
