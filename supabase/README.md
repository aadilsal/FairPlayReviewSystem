# Supabase Database Setup for FairPlay Review System

This directory contains all the SQL migration scripts and setup files for the FairPlay Review System database.

## 📁 Directory Structure

```
supabase/
├── migrations/
│   ├── 00001_create_users_table.sql
│   ├── 00002_create_matches_table.sql
│   ├── 00003_create_reviews_table.sql
│   └── 00004_create_notifications_table.sql
├── setup.sql          # Complete database setup (all-in-one)
├── seed.sql           # Sample data for testing
└── README.md          # This file
```

## 🚀 Quick Setup

### Option 1: Using setup.sql (Recommended)

1. Go to your [Supabase Dashboard](https://app.supabase.com)
2. Select your project: `jdefrtfaphntvpavxmhp`
3. Navigate to **SQL Editor** in the left sidebar
4. Click **New Query**
5. Copy and paste the entire contents of `setup.sql`
6. Click **Run** or press `Ctrl+Enter`

This will create all tables, indexes, constraints, and policies in one go.

### Option 2: Using Individual Migrations

Run each migration file in order:

1. `00001_create_users_table.sql`
2. `00002_create_matches_table.sql`
3. `00003_create_reviews_table.sql`
4. `00004_create_notifications_table.sql`

## 📊 Database Schema

### Tables

#### 1. **users**

Stores user account information.

```sql
- id (BIGSERIAL, PRIMARY KEY)
- username (VARCHAR, UNIQUE, NOT NULL)
- email (VARCHAR, UNIQUE, NOT NULL)
- password_hash (VARCHAR, NOT NULL)
- avatar (TEXT, NULLABLE)
- created_at (TIMESTAMP WITH TIME ZONE)
- updated_at (TIMESTAMP WITH TIME ZONE)
```

#### 2. **matches**

Stores cricket match information.

```sql
- id (BIGSERIAL, PRIMARY KEY)
- team_a (VARCHAR, NOT NULL)
- team_b (VARCHAR, NOT NULL)
- date (TIMESTAMP WITH TIME ZONE, NOT NULL)
- status (VARCHAR, NOT NULL) -- 'scheduled', 'in_progress', 'completed', 'cancelled', 'postponed'
- created_at (TIMESTAMP WITH TIME ZONE)
- updated_at (TIMESTAMP WITH TIME ZONE)
```

#### 3. **reviews**

Stores user reviews and AI analysis for matches.

```sql
- id (BIGSERIAL, PRIMARY KEY)
- match_id (BIGINT, FOREIGN KEY → matches.id)
- user_id (BIGINT, FOREIGN KEY → users.id)
- content (TEXT, NOT NULL)
- analysis (TEXT, NULLABLE)
- created_at (TIMESTAMP WITH TIME ZONE)
- updated_at (TIMESTAMP WITH TIME ZONE)
```

#### 4. **notifications**

Stores user notifications.

```sql
- id (BIGSERIAL, PRIMARY KEY)
- user_id (BIGINT, FOREIGN KEY → users.id)
- message (TEXT, NOT NULL)
- read (BOOLEAN, DEFAULT FALSE)
- created_at (TIMESTAMP WITH TIME ZONE)
- updated_at (TIMESTAMP WITH TIME ZONE)
```

## 🔐 Security Features

### Row Level Security (RLS)

All tables have RLS enabled with appropriate policies:

- **Users**: Can read all profiles, update own profile
- **Matches**: Public read access, authenticated write access
- **Reviews**: Public read access, users manage their own reviews
- **Notifications**: Users can only see and manage their own notifications

### Indexes

Performance-optimized indexes on:

- Foreign keys
- Frequently queried columns (email, username, date, status)
- Composite indexes for common queries

### Constraints

- Email validation regex
- Match status enum validation
- Foreign key constraints with CASCADE delete

## 🧪 Testing with Sample Data

To populate your database with test data:

1. After running `setup.sql`, run `seed.sql` in the SQL Editor
2. This will create:
   - 4 sample users (password: `password123`)
   - 7 sample matches
   - 5 sample reviews
   - 6 sample notifications

## 🔑 Environment Variables

Make sure your `.env` file has the correct Supabase credentials:

```env
SUPABASE_URL=https://jdefrtfaphntvpavxmhp.supabase.co
SUPABASE_KEY=sb_publishable_XzYbvyz7ZJSeku_rRuYy-A_QKlC0Y6u
```

## 📝 Verification

After running the setup, verify your tables:

```sql
-- Check if all tables exist
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
AND table_type = 'BASE TABLE';

-- Count records in each table
SELECT 'users' as table_name, COUNT(*) as count FROM public.users
UNION ALL
SELECT 'matches', COUNT(*) FROM public.matches
UNION ALL
SELECT 'reviews', COUNT(*) FROM public.reviews
UNION ALL
SELECT 'notifications', COUNT(*) FROM public.notifications;

-- Check foreign key constraints
SELECT
    tc.table_name,
    kcu.column_name,
    ccu.table_name AS foreign_table_name,
    ccu.column_name AS foreign_column_name
FROM information_schema.table_constraints AS tc
JOIN information_schema.key_column_usage AS kcu
    ON tc.constraint_name = kcu.constraint_name
JOIN information_schema.constraint_column_usage AS ccu
    ON ccu.constraint_name = tc.constraint_name
WHERE tc.constraint_type = 'FOREIGN KEY'
AND tc.table_schema = 'public';
```

## 🐛 Troubleshooting

### Error: "relation already exists"

- Tables already exist. Drop them first or use `DROP TABLE IF EXISTS` statements.

### Error: "permission denied"

- Ensure you're running queries as a user with sufficient privileges.
- Check RLS policies if data access fails.

### Connection Issues

- Verify your Supabase URL and API key in `.env`
- Check if your IP is allowed in Supabase settings
- Ensure Supabase project is not paused

## 🔄 Rollback

To drop all tables and start fresh:

```sql
-- Drop all tables (this will delete all data!)
DROP TABLE IF EXISTS public.notifications CASCADE;
DROP TABLE IF EXISTS public.reviews CASCADE;
DROP TABLE IF EXISTS public.matches CASCADE;
DROP TABLE IF EXISTS public.users CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;
```

## 📚 Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)

## 📧 Support

For issues or questions about the database setup, contact the development team.
