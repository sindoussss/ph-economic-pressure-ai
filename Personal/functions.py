def signup():
    global name
    name = input("Insert your name: ")
    global password
    password = input("Insert your password: ")
    repeat_password = input("Repeat your password: ")

    if password == repeat_password:
     print("Thank you for signing up")
     print("Login")
     login()
    else:
        while password != repeat_password:
            print("Passwords do not match try again")
            signup()

def login():
    login_name = input("Insert your name: ")
    login_password = input("Insert your password: ")

    if login_name == name and login_password == password:
        print("Login successful")
    while login_name != name and login_password != password:
        print("Incorrect password or Username \n")
        login()


choice = input("Do you want to sign up or log in? ")

if choice == "Sign up":
    signup()
elif choice == "Login":
    login()

