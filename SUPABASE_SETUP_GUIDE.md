# 🚀 Quick Start Guide - Supabase Database Setup

## Step-by-Step Instructions

### 1️⃣ Access Supabase SQL Editor

1. Open your browser and go to: https://app.supabase.com
2. Log in to your account
3. Select your project: `jdefrtfaphntvpavxmhp`
4. Click on **SQL Editor** in the left sidebar
5. Click **New Query** button

### 2️⃣ Run the Complete Setup

1. Open the file: `supabase/setup.sql`
2. Copy ALL the contents (Ctrl+A, Ctrl+C)
3. Paste into the Supabase SQL Editor
4. Click the **RUN** button (or press Ctrl+Enter)
5. Wait for completion (should show "Success. No rows returned")

✅ **All tables, indexes, and policies are now created!**

### 3️⃣ (Optional) Add Sample Data

If you want to test with sample data:

1. Click **New Query** in Supabase SQL Editor
2. Open the file: `supabase/seed.sql`
3. Copy and paste all contents
4. Click **RUN**

✅ **Sample users, matches, reviews, and notifications are now available!**

### 4️⃣ Verify Setup

Run this query in SQL Editor to verify:

```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE'
ORDER BY table_name;
```

You should see:

- ✓ matches
- ✓ notifications
- ✓ reviews
- ✓ users

### 5️⃣ Test Your API

Now you can test your FastAPI backend:

```powershell
# In your project directory (with venv activated)
cd "d:\Aadil Laptop\FAST\FYP\FairPlayReviewSystem"
uvicorn API.main_api:app --reload --host 0.0.0.0 --port 8000
```

Visit: http://localhost:8000/docs to see the interactive API documentation!

## 📋 Test Credentials

If you ran `seed.sql`, you can use these test accounts:

**User 1:**

- Email: `john@example.com`
- Password: `password123`

**User 2:**

- Email: `jane@example.com`
- Password: `password123`

**Admin:**

- Email: `admin@example.com`
- Password: `password123`

## 🔍 Check Table Contents

```sql
-- View all users
SELECT * FROM users;

-- View all matches
SELECT * FROM matches;

-- View reviews with details
SELECT
    r.content,
    u.username,
    m.team_a, m.team_b
FROM reviews r
JOIN users u ON r.user_id = u.id
JOIN matches m ON r.match_id = m.id;
```

## ⚠️ Important Notes

1. **RLS Policies**: All tables have Row Level Security enabled
2. **Passwords**: The sample passwords are hashed with bcrypt
3. **Production**: Change all passwords and secrets for production use
4. **API Key**: Use the publishable (anon) key in your frontend, service role key only in backend

## 🎯 Next Steps

1. ✅ Database is set up
2. ✅ API is configured
3. 🔜 Start your FastAPI server
4. 🔜 Test endpoints using /docs
5. 🔜 Connect your React Native frontend

## 🆘 Need Help?

If you encounter any issues:

1. Check the Supabase logs in the dashboard
2. Verify your .env file has correct credentials
3. Make sure all packages are installed: `pip install -r requirements.txt`
4. Check if the API server starts without errors

---

**You're all set! Your FairPlay Review System backend is ready! 🎉**
