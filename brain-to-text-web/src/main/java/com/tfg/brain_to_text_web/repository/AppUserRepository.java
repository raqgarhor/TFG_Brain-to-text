package com.tfg.brain_to_text_web.repository;

import com.tfg.brain_to_text_web.model.AppUser;
import java.util.Optional;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

public interface AppUserRepository extends JpaRepository<AppUser, Long> {

    @Query(value = """
            select *
            from app_users
            where username = :username
              and enabled = true
              and role = 'ADMIN'
              and password_hash = crypt(:password, password_hash)
            limit 1
            """, nativeQuery = true)
    Optional<AppUser> findValidAdmin(
            @Param("username") String username,
            @Param("password") String password
    );
}
