import random
import csv
import hashlib
from datetime import datetime, timedelta, timezone

random.seed(42)

# D1 Business Email Intent Dataset Generation
d1_classes = {
    "request": [
        ("Request for budget approval for Q3 marketing campaign", "Dear Manager, Please find attached the budget proposal for the Q3 campaign. Could you review and approve the requested funds by Friday? Best regards."),
        ("Access request to project analytics dashboard", "Hi IT Support, I need read-write access to the analytics dashboard for the new project. Please assign the appropriate permissions. Thanks!"),
        ("Request for software license renewal", "Hello Procurement, Our team requires 5 additional licenses for the IDE software. Please process this request at your earliest convenience."),
        ("Data export request for financial audit", "Hi Team, Can you please export the sales transaction records for the last fiscal year and send them over as a CSV file? Thank you."),
        ("Request for design assets for mobile app launch", "Dear Creative Team, We need the high-resolution logo files and icons for the upcoming mobile application release. Appreciate your help."),
        ("Equipment request: Dual monitor setup for new developer", "Hi Facilities, A new developer is joining our team next Monday. Could you please arrange a dual-monitor workstation setup at Desk 42? Thanks."),
        ("API documentation request for third-party integration", "Hello Engineering, Could you share the updated API specs and sample payloads for the payment gateway service? Best regards."),
        ("Leave request for upcoming conference", "Hi Lead, I would like to request leave from October 12 to October 15 to attend the annual AI developer conference. Please approve. Thanks!"),
    ],
    "meeting": [
        ("Schedule weekly sprint sync meeting", "Hi Team, Let us schedule our weekly sprint sync meeting for Thursday at 10 AM PST. Please confirm your availability or propose a new time."),
        ("Project kick-off meeting invitation", "Dear Colleagues, You are invited to the project kick-off call on Monday at 2 PM UTC. We will review project goals, deliverables, and timelines."),
        ("Rescheduling architecture review session", "Hi All, Due to a scheduling conflict, we need to move the architecture review session to Wednesday at 4 PM. Hope this works for everyone."),
        ("Follow-up meeting regarding client feedback", "Hello Team, Let us catch up tomorrow for 30 minutes to address the client feedback received yesterday. Calendar invite to follow."),
        ("Quarterly performance review meeting confirmation", "Hi Manager, Confirming our quarterly review meeting scheduled for tomorrow at 11 AM in Conference Room B."),
        ("Emergency sync on production deployment issue", "Team, Please join a quick 15-minute sync on Teams right now to discuss the production deployment failure."),
        ("Vendor product demonstration call", "Dear Procurement, The vendor has confirmed the product demo call for Friday at 3 PM. Please join to evaluate their offering."),
        ("One-on-one catchup meeting request", "Hi Alex, Do you have 20 minutes open this Thursday for a quick 1-on-1 catch-up regarding team allocation? Let me know!"),
    ],
    "complaint": [
        ("System downtime and latency issue in production database", "To Support, Our team has been experiencing extreme latency and frequent timeout errors when querying the production database. This is severely disrupting operations."),
        ("Unresolved ticket #4920: Login authentication failure", "Dear Helpdesk, I submitted ticket #4920 three days ago regarding login failures, but there has been zero response. This lack of communication is unacceptable."),
        ("Incorrect billing charge on monthly subscription invoice", "To Billing Department, We noticed an unauthorized surcharge on our monthly invoice #INV-8821. Please rectify this error immediately and issue a revised invoice."),
        ("Broken feature in latest software release v2.4", "Hi Product Team, The export feature completely crashes the application in v2.4. This broken update was released without adequate QA testing."),
        ("Poor service quality and delayed delivery of reports", "Dear Account Manager, The weekly analytics reports have been consistently delivered late for the past three weeks. We expect better reliability."),
        ("Mobile application crashing continuously on iOS 17", "To Support, Users are reporting that the app crashes immediately upon opening on iOS 17. Please prioritize fixing this bug."),
        ("Missing documentation for REST API endpoints", "Dear DevRel Team, The public documentation is completely missing description for authentication headers. It took hours of trial and error to integrate."),
        ("Network connectivity drops in North Wing office", "Hi IT Infra, The Wi-Fi connection in the North Wing keeps dropping every 10 minutes. Please investigate the access points."),
    ],
    "information": [
        ("FYI: Updated office security guidelines effective next month", "Dear All, Please note that updated office access procedures will take effect on the first of next month. Review the policy attached for details."),
        ("Company-wide holiday schedule for 2026", "Hello Everyone, Attached is the official company holiday calendar for the calendar year 2026. Keep this for your reference."),
        ("Release notes for platform update v3.1", "Team, Platform update v3.1 has been successfully deployed. Highlights include improved search indexing and security patches."),
        ("Minutes of meeting: Product strategy session", "Hi All, Here are the minutes from yesterday's product strategy session. No action required; for your information only."),
        ("New team member announcement: Welcome Sarah", "Dear Team, We are thrilled to welcome Sarah Jenkins as Senior Data Scientist. She brings 8 years of ML engineering experience."),
        ("Internal newsletter: Monthly engineering digest", "Hello Engineers, Check out the latest edition of our internal tech digest featuring articles on microservices and vector databases."),
        ("Notice of scheduled maintenance window this weekend", "All, Maintenance will be performed on the main server cluster this Saturday from 1 AM to 4 AM EST. Services may be intermittently unavailable."),
        ("Policy update regarding remote work expense reimbursement", "Dear Employees, The expense reimbursement policy for remote work equipment has been updated. Please consult the intranet portal."),
    ],
    "urgent_action": [
        ("CRITICAL: Production server down - Immediate action required", "URGENT: Main web application server is completely unresponsive. DevOps team must initiate disaster recovery protocols immediately!"),
        ("ALERT: Potential security breach detected on admin portal", "SECURITY ALERT: Multiple unauthorized root login attempts detected from unrecognized IP address. Lock down admin accounts immediately!"),
        ("ACTION REQUIRED: Contract renewal deadline expires today", "URGENT: The primary cloud vendor contract expires at midnight. Legal and procurement must sign and submit the renewal document before 5 PM!"),
        ("URGENT: Customer payment processing pipeline failing", "CRITICAL ERROR: Payment gateway integration is throwing 500 server errors for all checkout transactions. Immediate fix required!"),
        ("HIGH PRIORITY: Data compliance audit submission due in 2 hours", "URGENT: The regulatory compliance team requires signed data audit forms before 4 PM today. Please complete immediately."),
        ("CRITICAL API OUTAGE: Customer authentication service offline", "URGENT: OAuth service is failing globally. All customer logins are currently blocked. Escalating to P0 incident."),
        ("ACTION REQUIRED: Password reset forced due to credential compromise", "URGENT: All engineering staff must reset their corporate VPN credentials immediately following a credential leak alert."),
        ("CRITICAL BUG: Data loss issue identified in database migration script", "URGENT: Stop all database migration jobs immediately! A critical flaw in script v1.8 is dropping user transaction tables."),
    ],
    "spam": [
        ("CONGRATULATIONS! You have won $10,000,000 cash prize!", "Dear Lucky Winner, Claim your mega cash jackpot prize now by clicking this link and providing your credit card details immediately!"),
        ("Exclusive discount on cheap pharmaceuticals & online degrees!", "Buy discount medications and earn an accredited university diploma online in 24 hours without exams! Limited time offer!"),
        ("Urgent wire transfer request from CEO", "Hello, I am currently in a meeting and need you to urgently transfer $5,000 via gift cards or wire transfer to this bank account."),
        ("Double your Bitcoin in 24 hours with guaranteed returns!", "Invest in our revolutionary crypto automated bot and earn 200% daily ROI guaranteed. Click here to deposit funds!"),
        ("Low cost SEO services - Rank #1 on Google in 3 days!", "Dear Website Owner, We offer guaranteed top search engine rankings for only $49/month. Contact us now for a free quote!"),
        ("Get rich quick with work from home opportunity!", "Earn $500 per hour typing emails from home! No experience required! Sign up today and get an instant $100 bonus!"),
        ("Refinance your mortgage with 0% interest rate today!", "Pre-approved mortgage refinancing offer! Lower your monthly payments to zero! Click here to claim your quote!"),
        ("Unclaimed inheritance fund of $5.5 Million waiting for you", "Dear Friend, I am a barrister representing a deceased client with your surname. Contact me to transfer $5.5M to your account."),
    ]
}

d1_rows = []
email_id_counter = 1000
start_date = datetime(2026, 1, 1, tzinfo=timezone.utc)

for class_name, templates in d1_classes.items():
    # Generate 60 samples per class to total 360 rows
    for i in range(60):
        tmpl_subj, tmpl_body = templates[i % len(templates)]
        subj = tmpl_subj if i < len(templates) else f"{tmpl_subj} - Ref #{100+i}"
        body = tmpl_body if i < len(templates) else f"{tmpl_body} (Tracking code: TRK-{200+i})"
        email_id = f"D1_{email_id_counter}"
        email_id_counter += 1
        thread_id = f"TH_{100 + (i % 15)}"
        sender_grp = f"GRP_{ (i % 5) + 1 }"
        ts = (start_date + timedelta(hours=i*3, minutes=i*7)).isoformat()
        
        d1_rows.append({
            "email_id": email_id,
            "subject": subj,
            "body": body,
            "label": class_name,
            "dataset_id": "business_intent",
            "thread_id": thread_id,
            "sender_group": sender_grp,
            "timestamp": ts
        })

# D2 Enron Spam Dataset Generation (Binary: legitimate vs spam)
d2_classes = {
    "legitimate": [
        ("Enron North America gas trading schedule", "Here is the revised natural gas delivery schedule for the Houston pipeline hub for next week. Please review the volume numbers."),
        ("Meeting notes from risk management committee", "Attached are the summary notes and action items from yesterday's risk committee meeting. Let me know if any adjustments are needed."),
        ("Quarterly financial report review draft", "Vince, Please take a look at the attached draft for the Q2 financial summary before we present it to executive management."),
        ("Pipeline capacity allocation update", "The revised capacity allocation for the Western transmission segment has been finalized. See attached spreadsheet for details."),
        ("Energy trading model calibration parameters", "Hi Mark, We updated the volatility curves in the pricing model. The updated parameters have been saved to the shared drive."),
        ("Schedule change for Houston energy conference", "Hi All, The keynote panel at the Houston energy conference has been moved to 2 PM on Tuesday. Please update your itineraries."),
        ("Contract confirmation for physical power delivery", "Please find the executed contract confirmation for the July power delivery agreement attached. Thank you."),
        ("IT system maintenance notification for Enron intranet", "The internal trading database will undergo routine maintenance this Sunday from midnight to 4 AM. Expect brief downtime."),
    ],
    "spam": [
        ("Buy low price stocks now before explosive growth!", "Hot stock alert! Company XYZ is set to skyrocket 500% this week. Buy shares now before the press release!"),
        ("Refinance your home loan with super low rates!", "Save thousands on your mortgage! Pre-approved low interest home loan refinancing available immediately! Click now!"),
        ("Pharmacy online mega sale - 80% off prescription drugs!", "Order top quality prescription medications online without a prescription. Fast worldwide discreet shipping guaranteed!"),
        ("Earn easy money online working 2 hours a day!", "Make $3000 weekly from home! Simple online tasks, zero experience needed. Register now for immediate access!"),
        ("Casino bonus $500 free chip no deposit required!", "Play online slots and roulette with $500 free bonus cash! Instant payout, 100% secure platform! Claim today!"),
        ("Cheap software licenses: Windows, Office, Photoshop!", "Save up to 90% on OEM software downloads! Genuine keys, instant activation download link sent to your inbox!"),
        ("Increase your website traffic by 10x guaranteed!", "Drive millions of real targeted visitors to your site! Proven marketing strategy starting at only $29. Try it now!"),
        ("Urgent notification: Your account requires immediate verification", "Dear user, Your online banking account will be suspended within 24 hours unless you verify your credentials here now!"),
    ]
}

d2_rows = []
email_id_counter = 2000

for class_name, templates in d2_classes.items():
    # Generate 160 samples per class to total 320 rows
    for i in range(160):
        tmpl_subj, tmpl_body = templates[i % len(templates)]
        subj = tmpl_subj if i < len(templates) else f"{tmpl_subj} [{i}]"
        body = tmpl_body if i < len(templates) else f"{tmpl_body} (Ref ID: ENR-{500+i})"
        email_id = f"D2_{email_id_counter}"
        email_id_counter += 1
        thread_id = f"TH_ENR_{100 + (i % 20)}"
        sender_grp = f"GRP_ENR_{ (i % 4) + 1 }"
        ts = (start_date + timedelta(hours=i*4, minutes=i*11)).isoformat()
        
        d2_rows.append({
            "email_id": email_id,
            "subject": subj,
            "body": body,
            "label": class_name,
            "dataset_id": "enron_spam",
            "thread_id": thread_id,
            "sender_group": sender_grp,
            "timestamp": ts
        })

# Save D1 CSV
fieldnames = ["email_id", "subject", "body", "label", "dataset_id", "thread_id", "sender_group", "timestamp"]
with open("data/business_email_intent.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(d1_rows)

# Save D2 CSV
with open("data/enron_spam.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(d2_rows)

print(f"D1 generated: {len(d1_rows)} rows.")
print(f"D2 generated: {len(d2_rows)} rows.")
