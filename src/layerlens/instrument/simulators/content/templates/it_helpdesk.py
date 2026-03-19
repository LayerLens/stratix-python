"""IT helpdesk scenario templates.

Topics: Hardware_Issue, Security_Incident, Software_Install, VPN_Access, Password_Reset
"""

IT_HELPDESK_TEMPLATES = {
    "scenario": "it_helpdesk",
    "agent_names": ["IT_Support_Agent", "Security_Agent", "Network_Agent"],
    "system_prompts": {
        "IT_Support_Agent": (
            "You are an IT helpdesk agent. Help employees with hardware issues, "
            "software installations, and general IT support. Create tickets for "
            "issues that require on-site support. Always check the asset database "
            "before recommending hardware replacements."
        ),
        "Security_Agent": (
            "You are an IT security agent handling security incidents, access "
            "reviews, and compliance checks. Follow incident response procedures "
            "for any suspected security events."
        ),
        "Network_Agent": (
            "You are a network support agent. Help users with VPN connectivity, "
            "network access issues, and remote work setup."
        ),
    },
    "topics": {
        "Hardware_Issue": {
            "user_messages": [
                "My laptop screen is flickering and I can't work properly. It started this morning.",
                "The keyboard on my work laptop has several keys that aren't responding.",
                "My laptop battery only lasts about 30 minutes now. It used to last 6 hours.",
            ],
            "agent_responses": [
                "I'm sorry to hear about the screen issue. Let me check your device details and available options.",
                "I can see your laptop is a ThinkPad T14 Gen 3, still under warranty. I've created a hardware support ticket and our technician can replace the screen. Would tomorrow morning work for a desk-side visit?",
                "Given the battery health is at 23%, it's time for a replacement. I've ordered a new battery for your model and it should arrive within 2 business days. In the meantime, I can provide a loaner laptop.",
            ],
            "tools": {
                "Get_Asset_Info": {
                    "input": {"employee_id": "EMP-{id}", "asset_type": "laptop"},
                    "output": {
                        "asset_tag": "LAP-{id}",
                        "model": "ThinkPad T14 Gen 3",
                        "purchase_date": "2023-06-15",
                        "warranty_end": "2026-06-15",
                        "battery_health": 23,
                        "last_service": "2024-08-20",
                    },
                },
                "Create_Service_Ticket": {
                    "input": {"asset_tag": "LAP-{id}", "issue_type": "hardware", "priority": "high"},
                    "output": {"ticket_id": "TK-{id}", "assigned_to": "John Smith", "eta": "Next business day"},
                },
            },
        },
        "Security_Incident": {
            "user_messages": [
                "I think I clicked on a phishing link in an email. What should I do?",
                "I noticed unauthorized login attempts on my account from an unfamiliar location.",
                "I accidentally shared a file with sensitive data to an external email address.",
            ],
            "agent_responses": [
                "Thank you for reporting this immediately. Let me initiate our security incident response procedure.",
                "I've temporarily locked your account as a precaution and initiated a security scan. I need to ask you a few questions: Did you enter any credentials on the phishing page? When exactly did you click the link?",
                "I've quarantined your email and initiated a malware scan on your device. The security team has been alerted. For now, please don't click any other links and change your password from a different, trusted device.",
            ],
            "tools": {
                "Create_Security_Incident": {
                    "input": {"type": "phishing", "severity": "medium", "reporter": "EMP-{id}"},
                    "output": {
                        "incident_id": "SEC-{id}",
                        "status": "investigating",
                        "actions_taken": ["account_locked", "scan_initiated", "team_notified"],
                        "response_time_sla": "1 hour",
                    },
                },
                "Scan_Device": {
                    "input": {"asset_tag": "LAP-{id}", "scan_type": "full"},
                    "output": {
                        "scan_id": "SCAN-{id}",
                        "status": "clean",
                        "threats_found": 0,
                        "last_scan": "2024-12-15T10:30:00Z",
                    },
                },
            },
        },
        "Software_Install": {
            "user_messages": [
                "I need to install Python 3.12 for a new project. Can you approve it?",
                "Can I get Adobe Creative Suite installed on my workstation?",
                "I need Docker Desktop for our development environment setup.",
            ],
            "agent_responses": [
                "Let me check the software catalog and your access permissions for that installation.",
                "Python 3.12 is available in our approved software catalog. I've pushed the installation to your device. It should be available within 15 minutes after a restart.",
                "Adobe Creative Suite requires manager approval due to its license cost. I've submitted the request to your manager for approval. You'll receive an email once it's approved and the installation will be pushed automatically.",
            ],
            "tools": {
                "Check_Software_Catalog": {
                    "input": {"software_name": "Python", "version": "3.12"},
                    "output": {
                        "available": True,
                        "approved": True,
                        "license_type": "open_source",
                        "requires_approval": False,
                        "install_method": "auto_push",
                    },
                },
                "Push_Software_Install": {
                    "input": {"asset_tag": "LAP-{id}", "software_id": "SW-PY312", "silent": True},
                    "output": {"install_id": "INS-{id}", "status": "queued", "eta_minutes": 15},
                },
            },
        },
        "VPN_Access": {
            "user_messages": [
                "I can't connect to the VPN from home. It keeps timing out.",
                "I'm traveling internationally next week. Will the VPN work from overseas?",
                "The VPN disconnects every 30 minutes. It's disrupting my work.",
            ],
            "agent_responses": [
                "Let me check the VPN service status and your connection configuration.",
                "I can see the VPN service is operational. The timeout issue is usually caused by a firewall or ISP blocking the VPN port. Let me guide you through some troubleshooting steps.",
                "Our VPN works in most international locations. I've enabled the alternate connection profile for your account that uses port 443, which works better in countries with restrictive firewalls. You'll also need to install our updated VPN client.",
            ],
            "tools": {
                "Check_VPN_Status": {
                    "input": {"employee_id": "EMP-{id}"},
                    "output": {
                        "vpn_enabled": True,
                        "last_connection": "2024-12-14T17:00:00Z",
                        "profile": "standard",
                        "vpn_service_status": "operational",
                        "assigned_gateway": "vpn-us-east-1",
                    },
                },
                "Update_VPN_Profile": {
                    "input": {"employee_id": "EMP-{id}", "profile": "international", "port": 443},
                    "output": {"updated": True, "new_profile": "international", "config_pushed": True},
                },
            },
        },
        "Password_Reset": {
            "user_messages": [
                "I forgot my Active Directory password and I'm locked out of everything.",
                "My password expired and the self-service reset isn't working.",
                "I need to reset my password but I don't have access to my recovery email.",
            ],
            "agent_responses": [
                "I can help you reset your password. First, I need to verify your identity.",
                "I've verified your identity successfully. I'm resetting your Active Directory password now. You'll receive a temporary password via SMS that you'll need to change on first login.",
                "Since you don't have access to your recovery email, I'll use the backup verification method. I've sent a verification code to your registered phone number. Once verified, I can proceed with the reset.",
            ],
            "tools": {
                "Verify_Employee_Identity": {
                    "input": {"employee_id": "EMP-{id}", "method": "manager_verification"},
                    "output": {
                        "verified": True,
                        "method_used": "manager_verification",
                        "verifier": "Jane Doe (Manager)",
                    },
                },
                "Reset_AD_Password": {
                    "input": {"employee_id": "EMP-{id}", "delivery_method": "sms"},
                    "output": {
                        "reset_successful": True,
                        "temporary_password_sent": True,
                        "must_change_on_login": True,
                        "expiry_hours": 24,
                    },
                },
            },
        },
    },
}
