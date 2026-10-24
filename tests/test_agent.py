"""
Unit and Integration Tests: Autonomous Receptionist Agent & Plan-Act-Verify Tool Engine
FPGA-Based Multilingual AI Receptionist
"""

import os
import sys
import pytest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.tools import ReceptionistTools, ToolResult
from src.agent import ReceptionistAgent, ActionTraceStep, AgentSessionMemory


def test_tools_department_lookup():
    tools = ReceptionistTools()
    
    # English lookup
    res_en = tools.lookup_department("Engineering")
    assert res_en.success
    assert "3rd Floor" in res_en.data["floor"]
    assert res_en.data["room"] == "Room 304"
    assert res_en.tool_name == "lookup_department"

    # Tamil alias lookup
    res_ta = tools.lookup_department("பொறியியல்")
    assert res_ta.success
    assert res_ta.data["department_id"] == "engineering"

    # Hindi alias lookup
    res_hi = tools.lookup_department("मानव संसाधन")
    assert res_hi.success
    assert res_hi.data["department_id"] == "human_resources"

    # Non-existent department
    res_none = tools.lookup_department("Astronomy Lab")
    assert not res_none.success
    assert "available_departments" in res_none.data


def test_tools_check_office_hours():
    tools = ReceptionistTools()
    res = tools.check_office_hours()
    assert res.success
    assert "9:00 AM" in res.data["weekdays"]
    assert res.data["sunday"] == "Closed"


def test_tools_get_contact_details():
    tools = ReceptionistTools()
    
    # By department
    res_dept = tools.get_contact_details("finance")
    assert res_dept.success
    assert res_dept.data["extension"] == "ext. 303"

    # By officer name
    res_person = tools.get_contact_details("Priya Sharma")
    assert res_person.success
    assert "HR" in res_person.data["name"] or "Human Resources" in res_person.data["name"]


def test_tools_check_appointment():
    tools = ReceptionistTools()

    # Existing appointment
    res_found = tools.check_appointment(visitor_name="Saranya")
    assert res_found.success
    assert res_found.data["appointment_id"] == "APT-1001"
    assert res_found.data["host_name"] == "Dr. Ramesh Kumar"

    # Missing appointment
    res_missing = tools.check_appointment(visitor_name="Unknown Visitor 123")
    assert not res_missing.success
    assert res_missing.data["status"] == "not_found"


def test_tools_consequential_action_safety_gating():
    tools = ReceptionistTools()

    # Attempt booking without explicit confirmation -> must be gated
    res_unconfirmed = tools.schedule_appointment(
        visitor_name="John Doe",
        host_name="Dr. Ramesh Kumar",
        requested_time="Tomorrow 10:00 AM",
        confirmed=False
    )
    assert not res_unconfirmed.success
    assert res_unconfirmed.requires_confirmation
    assert res_unconfirmed.confirmation_prompt is not None

    # After explicit confirmation -> committed
    res_confirmed = tools.schedule_appointment(
        visitor_name="John Doe",
        host_name="Dr. Ramesh Kumar",
        requested_time="Tomorrow 10:00 AM",
        confirmed=True
    )
    assert res_confirmed.success
    assert not res_confirmed.requires_confirmation
    assert "APT-" in res_confirmed.data["appointment_id"]


def test_agent_goal_understanding_and_entities():
    agent = ReceptionistAgent()

    # Direction goal with department
    g1 = agent.understand_goal("Where is the engineering and fpga lab located?", "English")
    assert g1["goal"] == "get_directions"
    assert g1["entities"].get("department") == "engineering"

    # Office hours in Hindi
    g2 = agent.understand_goal("कार्यालय का समय क्या है?", "Hindi")
    assert g2["goal"] == "check_hours"

    # Appointment check with visitor name in Tamil
    g3 = agent.understand_goal("என் பெயர் சரண்யா எனக்கு சந்திப்பு உள்ளதா?", "Tamil")
    assert g3["goal"] == "check_appointment"
    assert g3["entities"].get("visitor_name") == "Saranya"

    # Emergency escalation
    g4 = agent.understand_goal("There is an emergency in the corridor, call a doctor immediately!", "English")
    assert g4["goal"] == "emergency_escalation"
    assert g4["urgency"] == "high"


def test_agent_plan_act_verify_loop_english():
    agent = ReceptionistAgent()

    reply, trace, meta = agent.run_agent_loop(
        visitor_text="What are your office opening hours?",
        language="English",
        use_llm=False
    )

    assert len(trace) >= 1
    assert trace[0].tool_selected == "check_office_hours"
    assert trace[0].tool_result_success
    assert "9:00 AM" in reply
    assert meta["goal"] == "check_hours"


def test_agent_plan_act_verify_loop_tamil():
    agent = ReceptionistAgent()

    reply, trace, meta = agent.run_agent_loop(
        visitor_text="பொறியியல் ஆய்வகம் எங்கு உள்ளது?",
        language="Tamil",
        use_llm=False
    )

    assert len(trace) >= 1
    assert trace[0].tool_selected == "get_campus_directions"
    assert trace[0].tool_result_success
    assert "3-வது தளம்" in reply or "304" in reply
    assert meta["language"] == "Tamil"


def test_agent_plan_act_verify_loop_hindi():
    agent = ReceptionistAgent()

    reply, trace, meta = agent.run_agent_loop(
        visitor_text="मानव संसाधन विभाग का कमरा नंबर क्या है?",
        language="Hindi",
        use_llm=False
    )

    assert len(trace) >= 1
    assert "201" in reply or "मानव संसाधन" in reply
    assert meta["language"] == "Hindi"


def test_agent_multi_turn_context_carryover():
    agent = ReceptionistAgent()

    # Turn 1: Discuss Engineering
    r1, t1, m1 = agent.run_agent_loop("Tell me about the engineering department", "English", use_llm=False)
    assert agent.memory.active_department == "engineering"

    # Turn 2: Follow-up using pronoun ("How do I get there?")
    r2, t2, m2 = agent.run_agent_loop("How do I get there?", "English", use_llm=False)
    assert t2[0].tool_selected == "get_campus_directions"
    assert t2[0].tool_input["to_department"] == "engineering"
    assert "3rd Floor" in r2 or "304" in r2

    # Clear memory
    agent.reset_session()
    assert agent.memory.active_department is None
    assert agent.memory.turn_count == 0


def test_agent_missing_info_clarification():
    agent = ReceptionistAgent()

    # Ask for appointment without specifying name
    reply, trace, meta = agent.run_agent_loop("Do I have an appointment today?", "English", use_llm=False)
    assert "full name" in reply.lower() or "provide your" in reply.lower()
    assert trace[0].tool_selected is None  # Pauses for clarification before executing tool
