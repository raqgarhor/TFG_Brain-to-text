package com.tfg.brain_to_text_web.repository;

import com.tfg.brain_to_text_web.model.Candidate;
import org.springframework.data.jpa.repository.JpaRepository;

public interface CandidateRepository extends JpaRepository<Candidate, Long> {
}
