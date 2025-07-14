from scrapper import run_scraper
from db import run_vector_db
from profiler import run_profiler
from report import run_report

if __name__ == "__main__":
    profile_url = input("Enter Reddit Profile URL: ").strip()
    if not profile_url:
        print("No URL provided.")
        exit(1)

    # Step 1: Scrape Reddit Profile
    username = run_scraper(profile_url)

    # Step 2: Create Vector Database
    run_vector_db(username)

    # Step 3: Generate User Persona
    run_profiler(username)

    # Step 4: Create PDF Report
    run_report(username)

    print(f"\n🎉 Completed! PDF report available as '{username}_persona_report.pdf'")
