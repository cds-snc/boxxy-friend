## Boxxy is your friend!
## Boxxy will imitate a user in interacting with web pages to perform a task.
test_url = "https://forms-staging.cdssandbox.xyz/en/form-builder"
local_path = "./models/gemma-4-E2B-it"

print("Welcome to Boxxy!")
print("----")
print("## --> Mode 1 : Boxxy will read HTML, and interact with the page based on the content.")
print("## --> Mode 2 : Boxxy will use Accessibility tags to mimic a screen reader.")
print("## --> Mode 3 : Boxxy will take screenshots of the page, and analyze the visual content then interact accordingly.")
print("----")
mode = input("What mode of Boxxy would you like to use? (1/2/3)")

if mode == "1":
    print("You have selected Mode 1: Boxxy will read HTML, and interact with the page based on the content.")
    from modules.mode1 import Mode1
    boxxy = Mode1(test_url, local_path)
elif mode == "2":
    print("You have selected Mode 2: Boxxy will use Accessibility tags to mimic a screen reader.")
    print("Sorry mode 2 isn't ready yet! Exiting!~")
    exit()
elif mode == "3":
    print("You have selected Mode 3: Boxxy will take screenshots of the page, and analyze the visual content then interact accordingly.")
    print("Sorry mode 3 isn't ready yet! Exiting!~")
    exit()
else:
    print("Invalid mode selected. Boxxy is sad now. :(")
    exit()

# Launch the Browser
boxxy.launch()

# Start exploring!~
boxxy.explore()

# Generate a report.
boxxy.report()