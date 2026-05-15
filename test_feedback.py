import asyncio
import json
from redis import asyncio as aioredis
from libs.schemas.feedback import FeedbackRequest
from apps.backend.services.feedback_collector import FeedbackCollector

async def test_feedback_collection():
    # Connect to Redis
    redis = await aioredis.from_url("redis://localhost:6379", decode_responses=True)
    collector = FeedbackCollector(redis)
    
    # Create sample feedback
    sample_feedback = FeedbackRequest(
        alert_id="alert_001",
        track_id=1,
        caption_sequence=["Person standing in restricted area", "Person reaching towards object"],
        original_label="Suspicious",
        human_label="Normal",
        human_note="This is an employee retrieving their bag",
        frame_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )
    
    # Store feedback
    record = await collector.store_feedback(sample_feedback)
    print(f"✓ Stored feedback for alert: {record.alert_id}")
    print(f"  Human label: {record.human_label}")
    print(f"  Timestamp: {record.timestamp}")
    
    # Add 2 more samples
    for i in range(2, 4):
        fb = FeedbackRequest(
            alert_id=f"alert_{i:03d}",
            track_id=i,
            caption_sequence=[f"Caption {i}-1", f"Caption {i}-2"],
            original_label="Alert",
            human_label="False Positive",
            human_note=f"Test feedback #{i}",
            frame_b64="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )
        await collector.store_feedback(fb)
        print(f"✓ Stored feedback for alert: alert_{i:03d}")
    
    # Retrieve feedback
    retrieved = await collector.get_feedback_by_alert_id("alert_001")
    print(f"\n✓ Retrieved alert_001: {retrieved.human_label}")
    
    # Get all feedback for track
    track_feedback = await collector.get_feedback_by_track_id(1)
    print(f"✓ Feedback records for track 1: {len(track_feedback)} records")
    
    await redis.close()

if __name__ == "__main__":
    asyncio.run(test_feedback_collection())