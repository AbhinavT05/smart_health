"""
FallGuard AI - Fall Confirmation & False Alarm Mitigation Lifecycle Tests
Proves:
1. Potential fall places FallEvent into PENDING_CONFIRMATION state.
2. "I'm OK" cancels emergency escalation without contacting emergency contacts.
3. "Need Help" or timeout confirms fall and escalates to MockEmergencyProvider.
4. Duplicate alerts prevented by cooldown.
"""

import asyncio
from datetime import datetime
from backend.database.session import SessionLocal
from backend.models.db_models import FallEvent, User
from backend.services.fall_confirmation_service import FallConfirmationService
from backend.services.alert_service import AlertService, MockEmergencyProvider

def test_fall_confirmation_user_cancels_im_ok():
    """Verify pressing 'I'm OK' cancels false alarms without triggering emergency alert."""
    async def _run():
        db = SessionLocal()
        mock_provider = MockEmergencyProvider()
        alert_service_inst = AlertService()
        alert_service_inst.set_provider(mock_provider)

        confirm_service = FallConfirmationService()

        try:
            owner = User(
                name="Okay Owner",
                email=f"okay_{datetime.now().timestamp()}@fallguard.ai",
                hashed_password="hash",
            )
            db.add(owner)
            db.commit()
            db.refresh(owner)
            # 1. Initiate pending confirmation with 10s timeout
            fall = await confirm_service.initiate_fall_confirmation(
                db=db,
                confidence=0.92,
                risk_level="HIGH",
                device_id="test_phone_ok",
                user_id=owner.id,
                ml_activity="WALKING",
                safety_override=True,
                timeout_seconds=10.0
            )

            assert fall.status == "PENDING_CONFIRMATION"
            assert fall.confirmed is False
            assert fall.alert_sent is False
            assert fall.id in confirm_service.pending_confirmations

            # 2. User presses "I'M OK"
            resolved_fall = await confirm_service.handle_user_response(
                db=db,
                fall_event_id=fall.id,
                action="I_AM_OK",
                acting_user_id=owner.id,
            )

            assert resolved_fall is not None
            assert resolved_fall.status == "CANCELLED_BY_USER"
            assert resolved_fall.confirmed is False
            assert resolved_fall.acknowledged is True
            assert fall.id not in confirm_service.pending_confirmations

            # 3. Assert NO emergency alerts were sent to emergency contact
            assert len(mock_provider.dispatched_alerts) == 0

        finally:
            db.close()

    asyncio.run(_run())

def test_fall_confirmation_user_escalates_need_help():
    """Verify pressing 'Need Help' immediately confirms fall and dispatches emergency alert."""
    async def _run():
        db = SessionLocal()
        mock_provider = MockEmergencyProvider()
        from backend.services.alert_service import alert_service
        alert_service.set_provider(mock_provider)

        confirm_service = FallConfirmationService()

        try:
            # Create user with emergency contact
            test_user = User(
                name="Alice Test",
                email=f"alice_{datetime.now().timestamp()}@fallguard.ai",
                hashed_password="hash",
                emergency_contact_name="Bob Caregiver",
                emergency_contact_phone="+15559988"
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)

            fall = await confirm_service.initiate_fall_confirmation(
                db=db,
                confidence=0.96,
                risk_level="HIGH",
                device_id="test_phone_help",
                user_id=test_user.id,
                ml_activity="FALL",
                safety_override=False,
                timeout_seconds=10.0
            )

            # User presses "NEED_HELP"
            resolved_fall = await confirm_service.handle_user_response(
                db=db,
                fall_event_id=fall.id,
                action="NEED_HELP",
                acting_user_id=test_user.id,
                location={"lat": 37.7749, "lon": -122.4194}
            )

            assert resolved_fall is not None
            assert resolved_fall.status == "CONFIRMED"
            assert resolved_fall.confirmed is True
            assert resolved_fall.acknowledged is True
            assert resolved_fall.acknowledged_by == "User (Smartphone)"
            assert resolved_fall.alert_sent is True

            # Emergency provider must have recorded dispatched alert
            assert len(mock_provider.dispatched_alerts) > 0
            last_alert = mock_provider.dispatched_alerts[-1]
            assert last_alert["contact_name"] == "Bob Caregiver"
            assert last_alert["contact_phone"] == "+15559988"
            assert last_alert["event"]["location"] == {"lat": 37.7749, "lon": -122.4194}

            # A client retry receives the settled status and does not dispatch twice.
            dispatched_count = len(mock_provider.dispatched_alerts)
            retry = await confirm_service.handle_user_response(
                db=db,
                fall_event_id=fall.id,
                action="NEED_HELP",
                acting_user_id=test_user.id,
            )
            assert retry.status == "CONFIRMED"
            assert len(mock_provider.dispatched_alerts) == dispatched_count

        finally:
            db.close()

    asyncio.run(_run())

def test_fall_confirmation_timeout_auto_escalation():
    """Verify confirmation timeout (e.g. 0.15s for test) automatically confirms and escalates."""
    async def _run():
        db = SessionLocal()
        mock_provider = MockEmergencyProvider()
        from backend.services.alert_service import alert_service
        alert_service.set_provider(mock_provider)

        confirm_service = FallConfirmationService()

        try:
            owner = User(
                name="Timeout Owner",
                email=f"timeout_{datetime.now().timestamp()}@fallguard.ai",
                hashed_password="hash",
                emergency_contact_name="Timeout Contact",
                emergency_contact_phone="+15551112222",
            )
            db.add(owner)
            db.commit()
            db.refresh(owner)
            # Fast 0.15s timeout
            fall = await confirm_service.initiate_fall_confirmation(
                db=db,
                confidence=0.94,
                risk_level="HIGH",
                device_id="test_phone_timeout",
                user_id=owner.id,
                timeout_seconds=0.15
            )

            # Wait for timer to expire
            await asyncio.sleep(0.3)

            db.refresh(fall)
            assert fall.status == "CONFIRMED"
            assert fall.confirmed is True
            assert fall.alert_sent is True
            assert len(mock_provider.dispatched_alerts) > 0

        finally:
            db.close()

    asyncio.run(_run())


def test_foreign_user_cannot_cancel_pending_fall_timeout():
    """A socket owner cannot cancel another user's pending confirmation timer."""
    async def _run():
        db = SessionLocal()
        confirm_service = FallConfirmationService()
        try:
            owner = User(name="Fall Owner", email=f"owner_{datetime.now().timestamp()}@fallguard.ai", hashed_password="hash")
            other = User(name="Other User", email=f"other_{datetime.now().timestamp()}@fallguard.ai", hashed_password="hash")
            db.add_all([owner, other])
            db.commit()
            db.refresh(owner)
            db.refresh(other)
            fall = await confirm_service.initiate_fall_confirmation(
                db=db, confidence=0.9, user_id=owner.id, device_id=f"owner_device_{owner.id}", timeout_seconds=10
            )

            rejected = await confirm_service.handle_user_response(
                db=db, fall_event_id=fall.id, action="I_AM_OK", acting_user_id=other.id
            )
            assert rejected is None
            assert fall.id in confirm_service.pending_confirmations
            assert not confirm_service.pending_confirmations[fall.id].task.done()

            accepted = await confirm_service.handle_user_response(
                db=db, fall_event_id=fall.id, action="I_AM_OK", acting_user_id=owner.id
            )
            assert accepted.status == "CANCELLED_BY_USER"
        finally:
            for pending in confirm_service.pending_confirmations.values():
                pending.task.cancel()
            db.close()
    asyncio.run(_run())
