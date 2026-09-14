from pathlib import Path
import random


# ============================================================
# TinyLuauGPT v5.2 Dataset Generator
# User-facing version: Beta 0.5
# ============================================================

OUTPUT_PATH = Path("dataset/luau_v52.txt")

TARGET_EXAMPLES = 30000

random.seed(52)


# ============================================================
# Helpers
# ============================================================

def block(mode, prompt, response):
    return (
        f"<{mode}>\n"
        f"{prompt}\n"
        f"<RESPONSE>\n"
        f"{response}\n"
        f"<END>\n"
    )


def choose(items):
    return random.choice(items)


# ============================================================
# Natural language variations
# ============================================================

player_phrases = [
    "플레이어가 게임에 들어오면",
    "플레이어가 접속하면",
    "새 플레이어가 들어오면",
    "게임에 플레이어가 참가하면",
    "유저가 게임에 들어왔을 때",
    "플레이어가 서버에 들어오면",
]

print_phrases = [
    "이름을 출력해줘.",
    "플레이어 이름을 콘솔에 보여줘.",
    "플레이어의 이름을 print로 출력해줘.",
    "닉네임을 출력하도록 해줘.",
]


# ============================================================
# CODE examples
# ============================================================

def code_examples():

    examples = []


    # --------------------------------------------------------
    # PlayerAdded
    # --------------------------------------------------------

    for p in player_phrases:
        for suffix in print_phrases:

            examples.append((
                f"{p} {suffix}",
                '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    print(player.Name)
end)'''
            ))


    # --------------------------------------------------------
    # PlayerRemoving
    # --------------------------------------------------------

    removing_prompts = [
        "플레이어가 게임을 나가면 이름을 출력해줘.",
        "유저가 서버를 떠날 때 이름을 출력해줘.",
        "플레이어가 나가면 콘솔에 이름을 표시해줘.",
        "플레이어가 게임을 종료하면 로그를 남겨줘.",
    ]

    for prompt in removing_prompts:

        examples.append((
            prompt,
            '''local Players = game:GetService("Players")

Players.PlayerRemoving:Connect(function(player)
    print(player.Name .. " left the game")
end)'''
        ))


    # --------------------------------------------------------
    # E key
    # --------------------------------------------------------

    key_prompts = [
        "E키를 누르면 Hello를 출력해줘.",
        "키보드에서 E를 누르면 메시지를 출력해줘.",
        "E 키 입력을 감지해서 Hello를 출력해줘.",
        "플레이어가 E키를 누르면 콘솔에 Hello를 출력해줘.",
    ]

    for prompt in key_prompts:

        examples.append((
            prompt,
            '''local UserInputService = game:GetService("UserInputService")

UserInputService.InputBegan:Connect(function(input, gameProcessed)
    if gameProcessed then
        return
    end

    if input.KeyCode == Enum.KeyCode.E then
        print("Hello")
    end
end)'''
        ))


    # --------------------------------------------------------
    # Part color
    # --------------------------------------------------------

    color_names = [
        ("빨간색", "Color3.fromRGB(255, 0, 0)"),
        ("파란색", "Color3.fromRGB(0, 0, 255)"),
        ("초록색", "Color3.fromRGB(0, 255, 0)"),
        ("노란색", "Color3.fromRGB(255, 255, 0)"),
        ("흰색", "Color3.fromRGB(255, 255, 255)"),
        ("검은색", "Color3.fromRGB(0, 0, 0)"),
        ("보라색", "Color3.fromRGB(170, 0, 255)"),
    ]

    for name, color in color_names:

        prompts = [
            f"Part를 {name}으로 바꿔줘.",
            f"파트의 색상을 {name}으로 설정해줘.",
            f"Part 색깔을 {name}으로 만들어줘.",
        ]

        for prompt in prompts:

            examples.append((
                prompt,
                f'''local part = workspace.Part

part.Color = {color}'''
            ))


    # --------------------------------------------------------
    # Part transparency
    # --------------------------------------------------------

    transparency_prompts = [
        "Part를 반투명하게 만들어줘.",
        "파트를 투명하게 만들어줘.",
        "Part의 Transparency를 0.5로 설정해줘.",
        "파트를 완전히 투명하게 만들어줘.",
    ]

    transparency_values = [
        0.5,
        1,
    ]

    for prompt in transparency_prompts:

        value = choose(transparency_values)

        examples.append((
            prompt,
            f'''local part = workspace.Part

part.Transparency = {value}'''
        ))


    # --------------------------------------------------------
    # Touched
    # --------------------------------------------------------

    touched_prompts = [
        "Part에 닿으면 Hello를 출력해줘.",
        "플레이어가 파트에 닿으면 메시지를 출력해줘.",
        "Part가 터치되면 콘솔에 Hello를 출력해줘.",
        "플레이어가 특정 Part를 만지면 실행되게 해줘.",
    ]

    for prompt in touched_prompts:

        examples.append((
            prompt,
            '''local part = workspace.Part

part.Touched:Connect(function(hit)
    print("Hello")
end)'''
        ))


    # --------------------------------------------------------
    # Humanoid
    # --------------------------------------------------------

    humanoid_prompts = [
        "플레이어 캐릭터가 죽으면 메시지를 출력해줘.",
        "플레이어가 죽었을 때 사망 메시지를 출력해줘.",
        "캐릭터가 죽는 것을 감지해줘.",
        "Humanoid가 죽으면 콘솔에 출력해줘.",
    ]

    for prompt in humanoid_prompts:

        examples.append((
            prompt,
            '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    player.CharacterAdded:Connect(function(character)
        local humanoid = character:WaitForChild("Humanoid")

        humanoid.Died:Connect(function()
            print("플레이어가 죽었습니다")
        end)
    end)
end)'''
        ))


    # --------------------------------------------------------
    # Walk speed
    # --------------------------------------------------------

    speed_prompts = [
        "플레이어의 걷는 속도를 20으로 설정해줘.",
        "캐릭터 이동 속도를 20으로 만들어줘.",
        "플레이어가 게임에 들어오면 WalkSpeed를 20으로 설정해줘.",
        "모든 플레이어의 이동 속도를 20으로 설정해줘.",
    ]

    for prompt in speed_prompts:

        examples.append((
            prompt,
            '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    player.CharacterAdded:Connect(function(character)
        local humanoid = character:WaitForChild("Humanoid")
        humanoid.WalkSpeed = 20
    end)
end)'''
        ))


    # --------------------------------------------------------
    # JumpPower
    # --------------------------------------------------------

    jump_prompts = [
        "플레이어의 점프력을 80으로 설정해줘.",
        "캐릭터가 더 높이 점프하도록 만들어줘.",
        "JumpPower를 80으로 설정해줘.",
    ]

    for prompt in jump_prompts:

        examples.append((
            prompt,
            '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    player.CharacterAdded:Connect(function(character)
        local humanoid = character:WaitForChild("Humanoid")
        humanoid.JumpPower = 80
    end)
end)'''
        ))


    # --------------------------------------------------------
    # WaitForChild
    # --------------------------------------------------------

    wait_prompts = [
        "캐릭터의 Humanoid를 안전하게 가져와줘.",
        "Humanoid가 생성될 때까지 기다려줘.",
        "캐릭터에서 Humanoid를 가져오는 코드를 만들어줘.",
    ]

    for prompt in wait_prompts:

        examples.append((
            prompt,
            '''local humanoid = character:WaitForChild("Humanoid")'''
        ))


    # --------------------------------------------------------
    # FindFirstChild
    # --------------------------------------------------------

    find_prompts = [
        "Part가 존재하는지 확인해줘.",
        "workspace에서 Part를 찾아줘.",
        "Part가 있는지 검사하는 코드를 만들어줘.",
    ]

    for prompt in find_prompts:

        examples.append((
            prompt,
            '''local part = workspace:FindFirstChild("Part")

if part then
    print("Part found")
end'''
        ))


    # --------------------------------------------------------
    # If statement
    # --------------------------------------------------------

    condition_prompts = [
        "돈이 100 이상이면 Hello를 출력해줘.",
        "점수가 50보다 크면 메시지를 출력해줘.",
        "값이 10과 같으면 출력해줘.",
        "숫자가 0보다 크면 실행해줘.",
    ]

    conditions = [
        '''if money >= 100 then
    print("Hello")
end''',

        '''if score > 50 then
    print("Hello")
end''',

        '''if value == 10 then
    print("Hello")
end''',

        '''if number > 0 then
    print("Hello")
end''',
    ]

    for prompt, code in zip(
        condition_prompts,
        conditions
    ):
        examples.append((
            prompt,
            code
        ))


    # --------------------------------------------------------
    # For loop
    # --------------------------------------------------------

    loop_prompts = [
        "1부터 10까지 출력해줘.",
        "숫자 1부터 10까지 반복해서 출력해줘.",
        "for문으로 1부터 10까지 출력해줘.",
        "10번 반복하는 코드를 만들어줘.",
    ]

    loop_codes = [
        '''for i = 1, 10 do
    print(i)
end''',

        '''for i = 1, 10 do
    print(i)
end''',

        '''for i = 1, 10 do
    print(i)
end''',

        '''for i = 1, 10 do
    print("Hello")
end''',
    ]

    for prompt, code in zip(
        loop_prompts,
        loop_codes
    ):
        examples.append((
            prompt,
            code
        ))


    # --------------------------------------------------------
    # Function
    # --------------------------------------------------------

    function_prompts = [
        "Hello를 출력하는 함수를 만들어줘.",
        "메시지를 출력하는 함수를 만들어줘.",
        "두 숫자를 더하는 함수를 만들어줘.",
        "간단한 Luau 함수를 만들어줘.",
    ]

    function_codes = [
        '''local function sayHello()
    print("Hello")
end

sayHello()''',

        '''local function showMessage()
    print("Hello")
end

showMessage()''',

        '''local function add(a, b)
    return a + b
end

print(add(5, 3))''',

        '''local function greet(name)
    print("Hello " .. name)
end

greet("Player")''',
    ]

    for prompt, code in zip(
        function_prompts,
        function_codes
    ):
        examples.append((
            prompt,
            code
        ))


    # --------------------------------------------------------
    # RemoteEvent
    # --------------------------------------------------------

    remote_prompts = [
        "RemoteEvent를 가져오는 코드를 만들어줘.",
        "ReplicatedStorage에서 RemoteEvent를 찾아줘.",
        "RemoteEvent를 사용하기 위한 기본 코드를 만들어줘.",
    ]

    for prompt in remote_prompts:

        examples.append((
            prompt,
            '''local ReplicatedStorage = game:GetService("ReplicatedStorage")

local remoteEvent = ReplicatedStorage:WaitForChild("RemoteEvent")'''
        ))


    # --------------------------------------------------------
    # Debounce
    # --------------------------------------------------------

    debounce_prompts = [
        "Part를 여러 번 빠르게 터치하지 못하게 해줘.",
        "터치 이벤트에 debounce를 넣어줘.",
        "같은 이벤트가 연속으로 실행되지 않게 해줘.",
    ]

    for prompt in debounce_prompts:

        examples.append((
            prompt,
            '''local part = workspace.Part

local debounce = false

part.Touched:Connect(function(hit)
    if debounce then
        return
    end

    debounce = true

    print("Touched")

    task.wait(1)

    debounce = false
end)'''
        ))


    return examples


# ============================================================
# EXPLAIN examples
# ============================================================

def explain_examples():

    return [

        (
            "local은 Luau에서 무엇을 하는 키워드야?",
            "local은 지역 변수를 만들 때 사용하는 키워드입니다."
        ),

        (
            "game:GetService는 무엇이야?",
            "game:GetService는 Roblox의 서비스를 가져올 때 사용하는 함수입니다."
        ),

        (
            "Players 서비스는 무엇을 관리해?",
            "Players 서비스는 게임에 참가한 플레이어와 관련된 기능을 관리합니다."
        ),

        (
            "PlayerAdded는 언제 실행돼?",
            "PlayerAdded 이벤트는 플레이어가 게임에 들어올 때 실행됩니다."
        ),

        (
            "PlayerRemoving은 언제 실행돼?",
            "PlayerRemoving 이벤트는 플레이어가 게임을 나갈 때 실행됩니다."
        ),

        (
            "Connect는 무엇을 하는 함수야?",
            "Connect는 이벤트가 발생했을 때 실행할 함수를 연결합니다."
        ),

        (
            "WaitForChild는 왜 사용해?",
            "WaitForChild는 지정한 객체가 생성될 때까지 기다릴 때 사용합니다."
        ),

        (
            "FindFirstChild는 무엇을 해?",
            "FindFirstChild는 지정한 이름의 자식 객체를 찾아 반환합니다."
        ),

        (
            "Humanoid는 무엇이야?",
            "Humanoid는 Roblox 캐릭터의 이동, 체력, 점프 등의 기능을 관리하는 객체입니다."
        ),

        (
            "Humanoid.Died는 무엇이야?",
            "Humanoid.Died는 캐릭터의 Humanoid가 죽었을 때 발생하는 이벤트입니다."
        ),

        (
            "Touched 이벤트는 언제 실행돼?",
            "Touched 이벤트는 BasePart가 다른 물체와 접촉했을 때 실행됩니다."
        ),

        (
            "WalkSpeed는 무엇이야?",
            "WalkSpeed는 Humanoid의 이동 속도를 설정하는 속성입니다."
        ),

        (
            "JumpPower는 무엇이야?",
            "JumpPower는 Humanoid의 점프 힘을 설정하는 속성입니다."
        ),

        (
            "if문은 왜 사용해?",
            "if문은 조건을 검사하고 조건이 참일 때 코드를 실행할 때 사용합니다."
        ),

        (
            "for문은 무엇을 해?",
            "for문은 특정 코드를 여러 번 반복해서 실행할 때 사용할 수 있습니다."
        ),

        (
            "function은 무엇이야?",
            "function은 여러 번 사용할 수 있는 코드 묶음을 만드는 데 사용합니다."
        ),

        (
            "RemoteEvent는 무엇이야?",
            "RemoteEvent는 클라이언트와 서버 사이에서 이벤트를 전달할 때 사용하는 객체입니다."
        ),

        (
            "debounce는 왜 사용해?",
            "debounce는 같은 이벤트가 짧은 시간에 반복 실행되는 것을 제한할 때 사용할 수 있습니다."
        ),

    ]


# ============================================================
# CHAT examples
# ============================================================

def chat_examples():

    return [

        ("안녕", "안녕하세요! 👋 저는 Roblox와 Luau 코딩을 도와주는 AI예요!"),
        ("안녕하세요", "안녕하세요! 😄 무엇을 만들어볼까요?"),
        ("ㅎㅇ", "ㅎㅇ! 😎 무엇을 도와드릴까요?"),
        ("고마워", "천만에요! 😄"),
        ("감사해", "도움이 되었다니 다행이에요! 🔥"),
        ("뭐 할 수 있어?", "Roblox Luau 코드 작성, 설명, 수정 등을 도와줄 수 있어요."),
        ("Roblox 코딩을 도와줄 수 있어?", "물론이죠! Roblox Luau 코딩을 함께 만들어볼 수 있어요."),
        ("코딩 시작하자", "좋아요! 🔥 만들고 싶은 기능을 말해주세요."),
        ("Luau 알아?", "네! Roblox에서 사용하는 Luau 코딩을 도와줄 수 있어요."),
        ("오늘 뭐 만들까?", "간단한 Roblox 시스템부터 만들어보는 건 어때요? 😎"),
    ]


# ============================================================
# FIX examples
# ============================================================

def fix_examples():

    return [

        (
            '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
print(player.Name)
end)''',

            '''local Players = game:GetService("Players")

Players.PlayerAdded:Connect(function(player)
    print(player.Name)
end)'''
        ),

        (
            '''local part = workspace.Part
part.Color = Color3.fromRGB(255, 0, 0)''',

            '''local part = workspace:WaitForChild("Part")

part.Color = Color3.fromRGB(255, 0, 0)'''
        ),

        (
            '''local function test()
print("Hello")
end''',

            '''local function test()
    print("Hello")
end'''
        ),

        (
            '''if value == 10
print("Hello")
end''',

            '''if value == 10 then
    print("Hello")
end'''
        ),

        (
            '''for i = 1, 10
print(i)
end''',

            '''for i = 1, 10 do
    print(i)
end'''
        ),

    ]


# ============================================================
# Generate
# ============================================================

def main():

    print("=" * 64)
    print("Luau AI v5.2 Dataset Generator")
    print("User-facing version: Beta 0.5")
    print("=" * 64)

    all_examples = []


    # Base examples
    code = code_examples()
    explain = explain_examples()
    chat = chat_examples()
    fix = fix_examples()


    # --------------------------------------------------------
    # Add base examples
    # --------------------------------------------------------

    for prompt, response in code:
        all_examples.append(
            block(
                "CODE",
                prompt,
                response
            )
        )


    for prompt, response in explain:
        all_examples.append(
            block(
                "EXPLAIN",
                prompt,
                response
            )
        )


    for prompt, response in chat:
        all_examples.append(
            block(
                "CHAT",
                prompt,
                response
            )
        )


    for prompt, response in fix:
        all_examples.append(
            block(
                "FIX",
                prompt,
                response
            )
        )


    # --------------------------------------------------------
    # Variation generation
    # --------------------------------------------------------

    # CODE examples get natural prompt variations
    for _ in range(TARGET_EXAMPLES * 3):

        prompt, response = choose(code)

        all_examples.append(
            block(
                "CODE",
                prompt,
                response
            )
        )


    # EXPLAIN
    for _ in range(TARGET_EXAMPLES):

        prompt, response = choose(explain)

        all_examples.append(
            block(
                "EXPLAIN",
                prompt,
                response
            )
        )


    # CHAT
    for _ in range(TARGET_EXAMPLES // 2):

        prompt, response = choose(chat)

        all_examples.append(
            block(
                "CHAT",
                prompt,
                response
            )
        )


    # FIX
    for _ in range(TARGET_EXAMPLES):

        prompt, response = choose(fix)

        all_examples.append(
            block(
                "FIX",
                prompt,
                response
            )
        )


    # --------------------------------------------------------
    # Shuffle
    # --------------------------------------------------------

    random.shuffle(
        all_examples
    )


    # --------------------------------------------------------
    # Limit
    # --------------------------------------------------------

    # Keep dataset around 30k examples
    all_examples = all_examples[
        :TARGET_EXAMPLES
    ]


    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    text = "".join(
        all_examples
    )


    OUTPUT_PATH.write_text(
        text,
        encoding="utf-8"
    )


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    mode_counts = {
        "CHAT": 0,
        "CODE": 0,
        "EXPLAIN": 0,
        "FIX": 0,
    }


    for item in all_examples:

        for mode in mode_counts:

            if item.startswith(
                f"<{mode}>"
            ):

                mode_counts[mode] += 1

                break


    print()
    print(
        "Examples:",
        f"{len(all_examples):,}"
    )

    print(
        "Characters:",
        f"{len(text):,}"
    )

    print()

    print(
        "CHAT    :",
        f"{mode_counts['CHAT']:,}"
    )

    print(
        "CODE    :",
        f"{mode_counts['CODE']:,}"
    )

    print(
        "EXPLAIN :",
        f"{mode_counts['EXPLAIN']:,}"
    )

    print(
        "FIX     :",
        f"{mode_counts['FIX']:,}"
    )

    print()

    print(
        "Output:",
        OUTPUT_PATH
    )

    print()

    print("=" * 64)
    print("v5.2 dataset generation complete!")
    print("User-facing version: Beta 0.5")
    print("=" * 64)


if __name__ == "__main__":
    main()