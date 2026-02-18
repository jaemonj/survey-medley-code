"""Questionnaire definitions shared across analyses."""

from __future__ import annotations

from typing import Dict, List


def define_questionnaires() -> Dict[str, List[str]]:
    """Return questionnaire-to-question-id mapping used across analyses."""
    grit_questions = ['Q01', 'Q02', 'Q03', 'Q04', 'Q05', 'Q06', 'Q07', 'Q08']
    brief_questions = [
        'Q09',
        'Q10',
        'Q11',
        'Q12',
        'Q13',
        'Q14',
        'Q15',
        'Q16',
        'Q17',
        'Q18',
        'Q19',
        'Q20',
        'Q21',
    ]
    future_time_questions = [
        'Q22',
        'Q23',
        'Q24',
        'Q25',
        'Q26',
        'Q27',
        'Q28',
        'Q29',
        'Q30',
        'Q31',
    ]
    upps_questions = ['Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37']
    impulsive_venture_questions = ['Q38', 'Q39', 'Q40']
    return {
        'grit': grit_questions,
        'brief': brief_questions,
        'future_time': future_time_questions,
        'upps': upps_questions,
        'impulsive_venture': impulsive_venture_questions,
    }
