#!/usr/bin/env python3
"""
Updated Database Setup Script for Smart Branch System with RAG Support

This script creates and initializes the SQLite database with all required tables
including the new username/password authentication fields and chat_history for RAG.
Run this before starting the main applications.
"""

import sqlite3
import os
import hashlib
from pathlib import Path
from datetime import datetime

def hash_password(password):
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def create_database():
    """Create and initialize the SQLite database"""
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Database file path
    db_path = "data/database.db"
    
    print(f"Creating database at: {db_path}")
    
    # Connect to database (creates file if doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable foreign key constraints
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create updated users table with username and password
    print("Creating users table with authentication...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT,
            phone TEXT,
            age INTEGER,
            gender TEXT,
            face_id TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create footfall table
    print("Creating footfall table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS footfall (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            entrance_count INTEGER DEFAULT 0,
            exit_count INTEGER DEFAULT 0,
            occupancy INTEGER DEFAULT 0,
            adults INTEGER DEFAULT 0,
            kids INTEGER DEFAULT 0,
            males INTEGER DEFAULT 0,
            females INTEGER DEFAULT 0
        )
    ''')
    
    # Create queue_entries table
    print("Creating queue_entries table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS queue_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            join_time DATETIME DEFAULT CURRENT_TIMESTAMP,
            estimated_wait INTEGER,
            actual_wait INTEGER,
            status TEXT DEFAULT 'waiting',
            position INTEGER,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create feedback table
    print("Creating feedback table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            rating INTEGER,
            comments TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create chat_history table for RAG chatbot
    print("Creating chat_history table for RAG chatbot...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create analytics table (for aggregated data)
    print("Creating analytics table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analytics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date DATE,
            total_entries INTEGER DEFAULT 0,
            total_exits INTEGER DEFAULT 0,
            peak_occupancy INTEGER DEFAULT 0,
            avg_wait_time INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create notifications log table
    print("Creating notifications_log table...")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS notifications_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            channel TEXT,
            message TEXT,
            status TEXT,
            sent_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Create indexes for better performance
    print("Creating indexes...")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_users_face_id ON users(face_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_footfall_timestamp ON footfall(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_queue_user_status ON queue_entries(user_id, status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_chat_timestamp ON chat_history(created_at)')
    
    # Commit changes
    conn.commit()
    
    # Insert sample data (optional)
    insert_sample_data = input("\nDo you want to insert sample data for testing? (y/n): ").lower() == 'y'
    if insert_sample_data:
        insert_sample_data_func(cursor, conn)
    
    # Display database info
    print_database_info(cursor)
    
    # Close connection
    conn.close()
    
    print(f"\nDatabase created successfully at: {os.path.abspath(db_path)}")
    print("\nYou can now run the applications:")
    print("  streamlit run app1.py  # Admin Dashboard")
    print("  streamlit run app2.py  # Customer Interface with RAG Chatbot")

def insert_sample_data_func(cursor, conn):
    """Insert sample data for testing"""
    print("Inserting sample data...")
    
    # Sample users with username/password authentication
    sample_users = [
        ('john_doe', hash_password('password123'), 'John Doe', 'john@email.com', '+1234567890', 30, 'male', 'sample_face_1'),
        ('jane_smith', hash_password('password123'), 'Jane Smith', 'jane@email.com', '+1234567891', 25, 'female', 'sample_face_2'),
        ('bob_johnson', hash_password('password123'), 'Bob Johnson', 'bob@email.com', '+1234567892', 35, 'male', 'sample_face_3'),
        ('alice_wonder', hash_password('demo123'), 'Alice Wonder', 'alice@email.com', '+1234567893', 28, 'female', None),
        ('demo_user', hash_password('demo'), 'Demo User', 'demo@smartbranch.com', '+1234567894', 30, 'other', None)
    ]
    
    # Insert users first
    cursor.executemany('''
        INSERT INTO users (username, password_hash, name, email, phone, age, gender, face_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', sample_users)
    
    # Commit users before inserting related data
    conn.commit()
    
    # Get the user IDs we just inserted
    cursor.execute('SELECT id FROM users ORDER BY id')
    user_ids = [row[0] for row in cursor.fetchall()]
    
    # Sample footfall data (last 7 days)
    from datetime import timedelta
    import random
    
    base_date = datetime.now() - timedelta(days=7)
    
    footfall_data = []
    for day in range(7):
        for hour in range(8, 20):  # Business hours 8 AM to 8 PM
            timestamp = base_date + timedelta(days=day, hours=hour)
            entrance_count = random.randint(5, 25)
            exit_count = random.randint(3, 23)
            occupancy = random.randint(20, 150)
            adults = random.randint(15, 120)
            kids = random.randint(0, 30)
            males = random.randint(10, 80)
            females = occupancy - males if males < occupancy else random.randint(5, 70)
            
            footfall_data.append((timestamp, entrance_count, exit_count, occupancy, adults, kids, males, females))
    
    cursor.executemany('''
        INSERT INTO footfall 
        (timestamp, entrance_count, exit_count, occupancy, adults, kids, males, females)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', footfall_data)
    
    # Sample feedback using actual user IDs
    sample_feedback = [
        (user_ids[0], 5, 'Great service, very quick!'),
        (user_ids[1], 4, 'Good experience overall'),
        (user_ids[2], 5, 'Excellent staff and short wait time'),
        (user_ids[3], 4, 'Nice app interface, easy to use'),
        (user_ids[4], 5, 'Perfect queue management system')
    ]
    
    cursor.executemany('''
        INSERT INTO feedback (user_id, rating, comments)
        VALUES (?, ?, ?)
    ''', sample_feedback)
    
    # Sample chat history using actual user IDs
    sample_chats = [
        (user_ids[0], 'How do I join the queue?', 'You can join the queue by clicking the "Join Queue" button in the Queue Status tab. You will receive your position and estimated wait time.'),
        (user_ids[1], 'How long will I wait?', 'Wait times are estimated at 5 minutes per person ahead of you in the queue. You will receive real-time updates on your position.'),
        (user_ids[2], 'Can I leave the queue?', 'Yes, you can leave the queue at any time by clicking the "Leave Queue" button. Your status will be updated immediately.'),
        (user_ids[0], 'How do notifications work?', 'You can receive notifications via email or SMS when it is almost your turn. You can configure your notification preferences in the Notifications tab.'),
        (user_ids[3], 'What is face recognition?', 'Face recognition is an optional login method that uses your photo to identify you. It creates a secure hash of your facial features for authentication.')
    ]
    
    cursor.executemany('''
        INSERT INTO chat_history (user_id, question, answer)
        VALUES (?, ?, ?)
    ''', sample_chats)
    
    conn.commit()
    print("Sample data inserted successfully!")
    print("\n" + "="*50)
    print("SAMPLE LOGIN CREDENTIALS")
    print("="*50)
    print("Username: john_doe       | Password: password123")
    print("Username: jane_smith     | Password: password123") 
    print("Username: bob_johnson    | Password: password123")
    print("Username: alice_wonder   | Password: demo123")
    print("Username: demo_user      | Password: demo")
    print("="*50)
    print("\nSample chat history has been added for testing the RAG chatbot!")

def print_database_info(cursor):
    """Print information about the created database"""
    print("\n" + "="*50)
    print("DATABASE INFORMATION")
    print("="*50)
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"Tables created: {len(tables)}")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        row_count = cursor.fetchone()[0]
        print(f"  - {table_name}: {row_count} rows")
    
    # Show users table structure
    cursor.execute("PRAGMA table_info(users)")
    columns = cursor.fetchall()
    print(f"\nUsers table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Show chat_history table structure
    cursor.execute("PRAGMA table_info(chat_history)")
    columns = cursor.fetchall()
    print(f"\nChat history table structure:")
    for col in columns:
        print(f"  - {col[1]} ({col[2]})")
    
    # Database file size
    db_path = "data/database.db"
    if os.path.exists(db_path):
        size_bytes = os.path.getsize(db_path)
        size_kb = size_bytes / 1024
        print(f"\nDatabase file size: {size_kb:.2f} KB")

def migrate_existing_database():
    """Migrate existing database to add username/password fields and chat_history table"""
    db_path = "data/database.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        print("Creating new database instead...")
        create_database()
        return
    
    print("Migrating existing database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if username column exists
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'username' not in columns:
        print("Adding username and password_hash columns...")
        
        # Add new columns
        cursor.execute('ALTER TABLE users ADD COLUMN username TEXT')
        cursor.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')
        
        # Update existing users with default values
        cursor.execute('SELECT id, name FROM users WHERE username IS NULL')
        users = cursor.fetchall()
        
        for user_id, name in users:
            # Generate username from name
            username = name.lower().replace(' ', '_').replace('-', '_')
            # Check if username exists
            counter = 1
            original_username = username
            cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            while cursor.fetchone():
                username = f"{original_username}_{counter}"
                counter += 1
                cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
            
            # Set default password
            default_password = hash_password('password123')
            
            cursor.execute('''
                UPDATE users SET username = ?, password_hash = ?
                WHERE id = ?
            ''', (username, default_password, user_id))
        
        print("Username/password migration completed!")
        
    # Check if chat_history table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'")
    if not cursor.fetchone():
        print("Adding chat_history table for RAG chatbot...")
        
        cursor.execute('''
            CREATE TABLE chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Add indexes
        cursor.execute('CREATE INDEX idx_chat_user ON chat_history(user_id)')
        cursor.execute('CREATE INDEX idx_chat_timestamp ON chat_history(created_at)')
        
        print("Chat history table created!")
    else:
        print("Chat history table already exists!")
    
    # Ensure all indexes exist
    required_indexes = [
        ('idx_users_username', 'users', 'username'),
        ('idx_chat_user', 'chat_history', 'user_id'),
        ('idx_chat_timestamp', 'chat_history', 'created_at')
    ]
    
    for index_name, table_name, column_name in required_indexes:
        cursor.execute(f"SELECT name FROM sqlite_master WHERE type='index' AND name='{index_name}'")
        if not cursor.fetchone():
            print(f"Creating index: {index_name}")
            cursor.execute(f'CREATE INDEX {index_name} ON {table_name}({column_name})')
    
    conn.commit()
    print("Migration completed successfully!")
    
    # Show updated users
    cursor.execute('SELECT username, name FROM users LIMIT 5')
    migrated_users = cursor.fetchall()
    if migrated_users:
        print("\nMigrated users (default password: 'password123'):")
        for username, name in migrated_users:
            print(f"  Username: {username} | Name: {name}")
    
    conn.close()

def check_database():
    """Check if database exists and show its contents"""
    db_path = "data/database.db"
    
    if not os.path.exists(db_path):
        print(f"Database not found at: {db_path}")
        print("Run this script to create it.")
        return
    
    print(f"Database found at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print_database_info(cursor)
    
    # Show sample users if any exist
    cursor.execute('SELECT username, name, email FROM users LIMIT 5')
    users = cursor.fetchall()
    if users:
        print(f"\nSample users:")
        for username, name, email in users:
            print(f"  @{username} - {name} ({email})")
    
    # Show sample chat history if exists
    cursor.execute('SELECT COUNT(*) FROM chat_history')
    chat_count = cursor.fetchone()[0]
    if chat_count > 0:
        print(f"\nChat history: {chat_count} conversations")
        cursor.execute('SELECT question FROM chat_history ORDER BY created_at DESC LIMIT 3')
        recent_questions = cursor.fetchall()
        print("Recent questions:")
        for question in recent_questions:
            print(f"  - {question[0][:60]}...")
    
    conn.close()

def reset_database():
    """Reset database by deleting and recreating it"""
    db_path = "data/database.db"
    
    if os.path.exists(db_path):
        confirm = input(f"This will delete the existing database at {db_path}. Continue? (y/n): ")
        if confirm.lower() == 'y':
            os.remove(db_path)
            print("Existing database deleted.")
        else:
            print("Operation cancelled.")
            return
    
    create_database()

def main():
    """Main function with menu options"""
    print("Smart Branch Database Setup with RAG Support")
    print("="*45)
    print("1. Create new database (with chat_history table)")
    print("2. Check existing database")
    print("3. Migrate existing database (add authentication + chat)")
    print("4. Reset database (delete and recreate)")
    print("5. Exit")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    if choice == '1':
        create_database()
    elif choice == '2':
        check_database()
    elif choice == '3':
        migrate_existing_database()
    elif choice == '4':
        reset_database()
    elif choice == '5':
        print("Goodbye!")
    else:
        print("Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()