package com.tfg.brain_to_text_web.controller;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

import com.tfg.brain_to_text_web.model.AppUser;
import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.service.AdminService;
import com.tfg.brain_to_text_web.service.TrialService;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageImpl;
import org.springframework.mock.web.MockHttpSession;
import org.springframework.ui.ConcurrentModel;
import org.springframework.web.servlet.mvc.support.RedirectAttributesModelMap;

@ExtendWith(MockitoExtension.class)
class AdminControllerTest {

    @Mock
    private AdminService adminService;

    @Mock
    private TrialService trialService;

    @InjectMocks
    private AdminController adminController;

    @Test
    void loginShowsLoginTemplateWhenUserIsNotAdmin() {
        assertEquals("admin-login", adminController.login(new MockHttpSession()));
    }

    @Test
    void doLoginStartsAdminSessionWhenCredentialsAreValid() {
        AppUser admin = new AppUser();
        admin.setDisplayName("Raquel");
        when(adminService.authenticateAdmin("admin", "admin123")).thenReturn(Optional.of(admin));
        MockHttpSession session = new MockHttpSession();

        String view = adminController.doLogin(
                "admin",
                "admin123",
                session,
                new RedirectAttributesModelMap()
        );

        assertEquals("redirect:/admin", view);
        assertEquals("Raquel", session.getAttribute("adminDisplayName"));
    }

    @Test
    void doLoginRedirectsToLoginWhenCredentialsAreInvalid() {
        when(adminService.authenticateAdmin("admin", "bad")).thenReturn(Optional.empty());
        RedirectAttributesModelMap redirectAttributes = new RedirectAttributesModelMap();

        String view = adminController.doLogin(
                "admin",
                "bad",
                new MockHttpSession(),
                redirectAttributes
        );

        assertEquals("redirect:/admin/login", view);
        assertEquals("Usuario o contraseña incorrectos.", redirectAttributes.getFlashAttributes().get("loginError"));
    }

    @Test
    void dashboardRedirectsToLoginWhenUserIsNotAdmin() {
        String view = adminController.dashboard(0, new MockHttpSession(), new ConcurrentModel());

        assertEquals("redirect:/admin/login", view);
        verify(trialService, never()).findTrialsPage(any(), any(Integer.class), any(Integer.class));
    }

    @Test
    void dashboardAddsStatsTrialsAndSplitsWhenUserIsAdmin() {
        MockHttpSession session = new MockHttpSession();
        session.setAttribute("adminUserId", 1L);
        session.setAttribute("adminDisplayName", "Raquel");
        Trial trial = new Trial();
        Page<Trial> page = new PageImpl<>(List.of(trial));
        AdminService.AdminStats stats = new AdminService.AdminStats(10, 20, 30, 1);
        when(trialService.findTrialsPage("all", 0, 10)).thenReturn(page);
        when(trialService.findSplits()).thenReturn(List.of("test", "val"));
        when(adminService.getStats()).thenReturn(stats);
        ConcurrentModel model = new ConcurrentModel();

        String view = adminController.dashboard(0, session, model);

        assertEquals("admin-dashboard", view);
        assertEquals("Raquel", model.getAttribute("adminName"));
        assertEquals(stats, model.getAttribute("stats"));
        assertEquals(List.of(trial), model.getAttribute("trials"));
        assertEquals(page, model.getAttribute("trialPage"));
        assertEquals(List.of("test", "val"), model.getAttribute("splits"));
    }

    @Test
    void createTrialRedirectsToLoginWhenUserIsNotAdmin() {
        String view = adminController.createTrial(
                "session",
                "val",
                "trial_0000",
                null,
                null,
                null,
                null,
                null,
                null,
                new MockHttpSession(),
                new RedirectAttributesModelMap()
        );

        assertEquals("redirect:/admin/login", view);
        verify(trialService, never()).createTrial(any(), any(), any(), any(), any(), any(), any(), any(), any());
    }

    @Test
    void createTrialDelegatesToServiceWhenUserIsAdmin() {
        MockHttpSession session = new MockHttpSession();
        session.setAttribute("adminUserId", 1L);
        RedirectAttributesModelMap redirectAttributes = new RedirectAttributesModelMap();

        String view = adminController.createTrial(
                "session",
                "val",
                "trial_0000",
                "sentence",
                "phonemes",
                "input",
                "logits",
                "image",
                "notes",
                session,
                redirectAttributes
        );

        assertEquals("redirect:/admin", view);
        verify(trialService).createTrial(
                "session",
                "val",
                "trial_0000",
                "sentence",
                "phonemes",
                "input",
                "logits",
                "image",
                "notes"
        );
        assertEquals("Ensayo creado correctamente.", redirectAttributes.getFlashAttributes().get("adminMessage"));
    }

    @Test
    void logoutRemovesAdminSessionAttributes() {
        MockHttpSession session = new MockHttpSession();
        session.setAttribute("adminUserId", 1L);
        session.setAttribute("adminDisplayName", "Raquel");

        String view = adminController.logout(session);

        assertEquals("redirect:/", view);
        assertEquals(null, session.getAttribute("adminUserId"));
        assertEquals(null, session.getAttribute("adminDisplayName"));
    }
}
