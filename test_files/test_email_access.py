# test_files/test_email_access.py

# Run with: python -m test_files.test_email_access

from bson import ObjectId
from configs.mongodb_config import get_database, workspace_id

def get_emails_for_workspace_test(workspace_id):
    """
    Test version of get_emails_for_workspace to check email retrieval.
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return []
    
    users_collection = db["users"]
    workspaces_collection = db["workspaces"]
    
    if not workspaces_collection or not users_collection:
        print("[Test] Database collections not available.")
        return []
    
    # Query workspaces collection for the workspace_id
    workspace_doc = workspaces_collection.find_one({"workspace_id": workspace_id})
    if not workspace_doc or "members" not in workspace_doc:
        print(f"[Test] Workspace {workspace_id} not found or has no members.")
        return []
    
    emails = []
    for member in workspace_doc["members"]:
        user_id = member.get("user")  # Assuming 'user' field holds user ID
        if user_id:
            # Query users collection for email
            user_doc = users_collection.find_one({"_id": user_id})  # Assuming _id is the user ID
            if user_doc and "email" in user_doc:
                emails.append(user_doc["email"])
            else:
                print(f"[Test] User {user_id} not found or has no email.")
    
    return emails

def print_all_users():
    """
    Print all documents in the 'users' collection.
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return
    
    users_collection = db["users"]
    print("[Test] Fetching all documents from 'users' collection...")
    
    documents = list(users_collection.find())
    if documents:
        print(f"[Test] Found {len(documents)} document(s):")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- User {i} ---")
            for key, value in doc.items():
                print(f"{key}: {value}")
    else:
        print("[Test] No documents found in 'users' collection.")

def print_all_workspaces():
    """
    Print all documents in the 'workspaces' collection.
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return
    
    workspaces_collection = db["workspaces"]
    print("[Test] Fetching all documents from 'workspaces' collection...")
    
    documents = list(workspaces_collection.find())
    if documents:
        print(f"[Test] Found {len(documents)} document(s):")
        for i, doc in enumerate(documents, 1):
            print(f"\n--- Document {i} ---")
            for key, value in doc.items():
                print(f"{key}: {value}")
    else:
        print("[Test] No documents found in 'workspaces' collection.")

def inspect_workspace_by_id(workspace_oid):
    """
    Fetch and display full workspace document from MongoDB using _id.
    
    workspace_oid: str or ObjectId of the workspace
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return
    
    workspaces_collection = db["workspaces"]

    # Ensure workspace_oid is ObjectId
    if isinstance(workspace_oid, str):
        try:
            workspace_oid = ObjectId(workspace_oid)
        except Exception as e:
            print(f"[Test] Invalid ObjectId string: {workspace_oid}")
            return

    workspace_doc = workspaces_collection.find_one({"_id": workspace_oid})
    
    if workspace_doc:
        print("\n===== Workspace Document =====")
        for key, value in workspace_doc.items():
            print(f"{key}: {value}")
        print("===============================\n")
    else:
        print(f"[Test] Workspace with _id {workspace_oid} not found.")

def inspect_workspace_with_emails(workspace_oid):
    """
    Fetch workspace document by _id and print member emails.
    
    workspace_oid: str or ObjectId of the workspace
    """
    db = get_database()
    if not db:
        print("[Test] Failed to connect to database.")
        return

    workspaces_collection = db["workspaces"]
    users_collection = db["users"]

    # Convert to ObjectId if needed
    if isinstance(workspace_oid, str):
        try:
            workspace_oid = ObjectId(workspace_oid)
        except Exception as e:
            print(f"[Test] Invalid ObjectId string: {workspace_oid}")
            return

    workspace_doc = workspaces_collection.find_one({"_id": workspace_oid})

    if not workspace_doc:
        print(f"[Test] Workspace with _id {workspace_oid} not found.")
        return

    print("\n===== Workspace Document =====")
    for key, value in workspace_doc.items():
        print(f"{key}: {value}")
    print("===============================\n")

    # Get member emails
    members = workspace_doc.get("members", [])
    if not members:
        print("[Test] Workspace has no members.")
        return

    print("[Test] Members and their emails:")
    for member in members:
        user_id = member.get("user")
        if not user_id:
            continue

        user_doc = users_collection.find_one({"_id": user_id})
        if user_doc and "email" in user_doc:
            print(f"- {user_doc.get('name')} ({user_doc.get('email')}) | Role: {member.get('role')}")
        else:
            print(f"- User with _id {user_id} not found or has no email")


if __name__ == "__main__":    
    print("\n[Test] Printing all users...")
    print_all_users()

    print("\n[Test] Printing all workspaces...")
    print_all_workspaces()

    print(f"[Test] Inspecting workspace_id: {workspace_id}")
    inspect_workspace_by_id(workspace_id)

    print(f"[Test] Inspecting workspace with _id: {workspace_id}")
    inspect_workspace_with_emails(workspace_id)

    # print(f"[Test] Checking emails for workspace_id: {workspace_id}")
    # emails = get_emails_for_workspace_test(workspace_id)
    # if emails:
    #     print(f"[Test] Found emails: {emails}")
    # else:
    #     print("[Test] No emails found. Check MongoDB data in 'workspaces' and 'users' collections.")