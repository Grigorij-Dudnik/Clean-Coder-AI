from datetime import datetime
from db import db
from fastapi import HTTPException


def create_invitation(from_uuid: str, to_uuid: str):
    """
    Insert an 'invitations' doc with status='new'.
    Prevents duplicate invitations between the same users.
    """
    invitations_coll = db["invitations"]

    # Check if invitation already exists
    existing_invitation = invitations_coll.find_one(
        {
            "from_uuid": from_uuid,
            "to_uuid": to_uuid,
            "status": {"$in": ["new", "accepted"]},  # Only check active invitations
        }
    )

    if existing_invitation:
        raise HTTPException(status_code=400, detail="Invitation already exists between these users")

    now = datetime.utcnow()
    invitation = {"from_uuid": from_uuid, "to_uuid": to_uuid, "status": "new", "created_at": now, "updated_at": now}
    invitations_coll.insert_one(invitation)
    return {"message": "Invitation created"}


def get_sent_invitations(current_user):
    """
    Return docs from 'invitations' where current_user is from_uuid,
    including profile data of the recipient.
    """
    invitations_coll = db["invitations"]

    pipeline = [
        {"$match": {"from_uuid": current_user["uuid"]}},
        {
            "$lookup": {
                "from": "campaigns" if current_user["role"] == "intern" else "interns",
                "localField": "to_uuid",
                "foreignField": "uuid",
                "as": "profile_data",
            }
        },
        {"$unwind": "$profile_data"},
        {
            "$project": {
                "_id": 0,
                "uuid": "$profile_data.uuid",
                "full_name": "$profile_data.full_name",
                "short_bio": "$profile_data.short_bio",
                "invitation_status": "$status",
            }
        },
    ]

    return list(invitations_coll.aggregate(pipeline))


def get_received_invitations(current_user):
    """
    Return docs from 'invitations' where current_user is to_uuid,
    including profile data of the sender.
    """
    invitations_coll = db["invitations"]

    pipeline = [
        {"$match": {"to_uuid": current_user["uuid"]}},
        {
            "$lookup": {
                "from": "campaigns" if current_user["role"] == "intern" else "interns",
                "localField": "from_uuid",
                "foreignField": "uuid",
                "as": "profile_data",
            }
        },
        {"$unwind": "$profile_data"},
        {
            "$project": {
                "_id": 0,
                "uuid": "$profile_data.uuid",
                "full_name": "$profile_data.full_name",
                "short_bio": "$profile_data.short_bio",
                "invitation_status": "$status",
            }
        },
    ]

    return list(invitations_coll.aggregate(pipeline))


def accept_invitation(invitation_id, current_user):
    """
    Accept an invitation if the current user is the recipient.
    """
    invitations_coll = db["invitations"]
    invitation = invitations_coll.find_one({"_id": invitation_id, "to_uuid": current_user["uuid"]})

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or not authorized to accept.")

    invitations_coll.update_one(
        {"_id": invitation_id}, {"$set": {"status": "accepted", "updated_at": datetime.utcnow()}}
    )
    return {"message": "Invitation accepted"}


def reject_invitation(invitation_id, current_user):
    """
    Reject an invitation if the current user is the recipient.
    """
    invitations_coll = db["invitations"]
    invitation = invitations_coll.find_one({"_id": invitation_id, "to_uuid": current_user["uuid"]})

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or not authorized to reject.")

    invitations_coll.update_one(
        {"_id": invitation_id}, {"$set": {"status": "rejected", "updated_at": datetime.utcnow()}}
    )
    return {"message": "Invitation rejected"}


def cancel_invitation(invitation_id, current_user):
    """
    Cancel an invitation if the current user is the sender.
    """
    invitations_coll = db["invitations"]
    invitation = invitations_coll.find_one({"_id": invitation_id, "from_uuid": current_user["uuid"]})

    if not invitation:
        raise HTTPException(status_code=404, detail="Invitation not found or not authorized to cancel.")

    invitations_coll.update_one(
        {"_id": invitation_id}, {"$set": {"status": "cancelled", "updated_at": datetime.utcnow()}}
    )
    return {"message": "Invitation cancelled"}
