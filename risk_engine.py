"""
Risk Scoring Engine for Eagle Surveillance
Author: Bhagyashri
Issue: #19 - Multi-factor risk scoring algorithm
"""

class RiskAnalyzer:
    def __init__(self):
        # Step 1: Assign weights to different risk factors
        # Yeh weights decide karenge ki kaunsi cheez kitni dangerous hai
        self.weights = {
            "danger_zone": 50.0,       # Sabse highest risk
            "restricted_zone": 30.0,   # Medium risk
            "suspicious_item": 20.0,   # Like backpacks in secure areas
            "base_person_risk": 10.0   # Normal presence
        }

    def calculate_risk_score(self, label: str, zones_present: list) -> float:
        """
        Calculates a risk score from 0 to 100 based on the detected object 
        and the zone it is currently in.
        """
        total_risk = 0.0

        # Rule 1: Check the Zone (Location mapping)
        # Agar object kisi unauthorized zone mein hai toh score badha do
        if "danger" in zones_present:
            total_risk += self.weights["danger_zone"]
        elif "restricted" in zones_present:
            total_risk += self.weights["restricted_zone"]

        # Rule 2: Check for Suspicious Items
        # Agar koi bag ya suitcase aisi jagah hai jahan nahi hona chahiye
        if label in ["backpack", "handbag", "suitcase"]:
            total_risk += self.weights["suspicious_item"]

        # Rule 3: Base Risk for Human Presence
        if label == "person":
            total_risk += self.weights["base_person_risk"]

        # Final Rule: Normalize the score
        # Make sure the score never goes beyond 100%
        final_score = min(total_risk, 100.0)
        
        return final_score

# --- Local Testing (To check if our logic works) ---
if __name__ == "__main__":
    analyzer = RiskAnalyzer()
    
    # Test Scenario: A person carrying a backpack in a restricted zone
    print("--- Eagle Risk Engine Test ---")
    
    # Test 1: Just a person in a safe zone
    score_1 = analyzer.calculate_risk_score("person", ["safe"])
    print(f"Test 1 (Person in Safe Zone) Risk Score: {score_1}/100")
    
    # Test 2: Backpack in restricted zone
    score_2 = analyzer.calculate_risk_score("backpack", ["restricted"])
    print(f"Test 2 (Backpack in Restricted Zone) Risk Score: {score_2}/100")
    
    # Test 3: Person in danger zone
    score_3 = analyzer.calculate_risk_score("person", ["danger"])
    print(f"Test 3 (Person in Danger Zone) Risk Score: {score_3}/100")