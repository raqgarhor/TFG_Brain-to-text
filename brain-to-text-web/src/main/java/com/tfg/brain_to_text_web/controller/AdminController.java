package com.tfg.brain_to_text_web.controller;

import com.tfg.brain_to_text_web.model.AppUser;
import com.tfg.brain_to_text_web.model.Trial;
import com.tfg.brain_to_text_web.service.AdminService;
import com.tfg.brain_to_text_web.service.TrialService;
import jakarta.servlet.http.HttpSession;
import java.util.List;
import org.springframework.data.domain.Page;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.servlet.mvc.support.RedirectAttributes;

@Controller
public class AdminController {

    private static final String ADMIN_ID = "adminUserId";
    private static final String ADMIN_NAME = "adminDisplayName";

    private final AdminService adminService;
    private final TrialService trialService;

    public AdminController(AdminService adminService, TrialService trialService) {
        this.adminService = adminService;
        this.trialService = trialService;
    }

    @GetMapping("/admin/login")
    public String login(HttpSession session) {
        if (isAdmin(session)) {
            return "redirect:/admin";
        }
        return "admin-login";
    }

    @PostMapping("/admin/login")
    public String doLogin(
            @RequestParam String username,
            @RequestParam String password,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        return adminService.authenticateAdmin(username, password)
                .map(admin -> startAdminSession(admin, session))
                .orElseGet(() -> {
                    redirectAttributes.addFlashAttribute("loginError", "Usuario o contraseña incorrectos.");
                    return "redirect:/admin/login";
                });
    }

    @PostMapping("/admin/logout")
    public String logout(HttpSession session) {
        session.removeAttribute(ADMIN_ID);
        session.removeAttribute(ADMIN_NAME);
        return "redirect:/";
    }

    @GetMapping("/admin")
    public String dashboard(
            @RequestParam(defaultValue = "0") int page,
            HttpSession session,
            Model model
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        Page<Trial> trialPage = trialService.findTrialsPage("all", page, 10);
        List<String> splits = trialService.findSplits();

        model.addAttribute("adminName", session.getAttribute(ADMIN_NAME));
        model.addAttribute("stats", adminService.getStats());
        model.addAttribute("trials", trialPage.getContent());
        model.addAttribute("trialPage", trialPage);
        model.addAttribute("splits", splits);
        return "admin-dashboard";
    }

    @GetMapping("/admin/trials/{id}/edit")
    public String editTrial(
            @PathVariable Long id,
            HttpSession session,
            Model model
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        Trial trial = trialService.findTrial(id);
        model.addAttribute("adminName", session.getAttribute(ADMIN_NAME));
        model.addAttribute("trial", trial);
        model.addAttribute("predictions", adminService.findPredictionsForTrial(id));
        return "admin-trial-edit";
    }

    @PostMapping("/admin/trials")
    public String createTrial(
            @RequestParam String sessionName,
            @RequestParam String split,
            @RequestParam String trialKey,
            @RequestParam(required = false) String realSentence,
            @RequestParam(required = false) String realPhonemes,
            @RequestParam(required = false) String inputShape,
            @RequestParam(required = false) String logitsShape,
            @RequestParam(required = false) String signalImagePath,
            @RequestParam(required = false) String notes,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            trialService.createTrial(
                    sessionName,
                    split,
                    trialKey,
                    realSentence,
                    realPhonemes,
                    inputShape,
                    logitsShape,
                    signalImagePath,
                    notes
            );
            redirectAttributes.addFlashAttribute("adminMessage", "Ensayo creado correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo crear el ensayo. Revisa que no exista ya.");
        }
        return "redirect:/admin";
    }

    @PostMapping("/admin/trials/{id}")
    public String updateTrial(
            @PathVariable Long id,
            @RequestParam String sessionName,
            @RequestParam String split,
            @RequestParam String trialKey,
            @RequestParam(required = false) String realSentence,
            @RequestParam(required = false) String realPhonemes,
            @RequestParam(required = false) String inputShape,
            @RequestParam(required = false) String logitsShape,
            @RequestParam(required = false) String signalImagePath,
            @RequestParam(required = false) String notes,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            trialService.updateTrial(
                    id,
                    sessionName,
                    split,
                    trialKey,
                    realSentence,
                    realPhonemes,
                    inputShape,
                    logitsShape,
                    signalImagePath,
                    notes
            );
            redirectAttributes.addFlashAttribute("adminMessage", "Ensayo actualizado correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo actualizar el ensayo.");
        }
        return "redirect:/admin/trials/" + id + "/edit";
    }

    @PostMapping("/admin/trials/{id}/delete")
    public String deleteTrial(
            @PathVariable Long id,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        trialService.deleteTrial(id);
        redirectAttributes.addFlashAttribute("adminMessage", "Ensayo eliminado correctamente.");
        return "redirect:/admin";
    }

    @PostMapping("/admin/trials/{trialId}/predictions")
    public String createPrediction(
            @PathVariable Long trialId,
            @RequestParam String modelName,
            @RequestParam String modelLabel,
            @RequestParam(required = false) String predictedPhonemes,
            @RequestParam(required = false) String predictedText,
            @RequestParam(required = false) String partialText,
            @RequestParam(required = false) String perValue,
            @RequestParam(required = false) String werValue,
            @RequestParam(required = false) String checkpointPer,
            @RequestParam(required = false) String notes,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            adminService.createPrediction(
                    trialId,
                    modelName,
                    modelLabel,
                    predictedPhonemes,
                    predictedText,
                    partialText,
                    perValue,
                    werValue,
                    checkpointPer,
                    notes
            );
            redirectAttributes.addFlashAttribute("adminMessage", "Predicción creada correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo crear la predicción. Revisa si ya existe para ese modelo.");
        }
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    @PostMapping("/admin/trials/{trialId}/predictions/{predictionId}")
    public String updatePrediction(
            @PathVariable Long trialId,
            @PathVariable Long predictionId,
            @RequestParam String modelName,
            @RequestParam String modelLabel,
            @RequestParam(required = false) String predictedPhonemes,
            @RequestParam(required = false) String predictedText,
            @RequestParam(required = false) String partialText,
            @RequestParam(required = false) String perValue,
            @RequestParam(required = false) String werValue,
            @RequestParam(required = false) String checkpointPer,
            @RequestParam(required = false) String notes,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            adminService.updatePrediction(
                    predictionId,
                    modelName,
                    modelLabel,
                    predictedPhonemes,
                    predictedText,
                    partialText,
                    perValue,
                    werValue,
                    checkpointPer,
                    notes
            );
            redirectAttributes.addFlashAttribute("adminMessage", "Predicción actualizada correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo actualizar la predicción.");
        }
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    @PostMapping("/admin/trials/{trialId}/predictions/{predictionId}/delete")
    public String deletePrediction(
            @PathVariable Long trialId,
            @PathVariable Long predictionId,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        adminService.deletePrediction(predictionId);
        redirectAttributes.addFlashAttribute("adminMessage", "Predicción eliminada correctamente.");
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    @PostMapping("/admin/trials/{trialId}/predictions/{predictionId}/candidates")
    public String createCandidate(
            @PathVariable Long trialId,
            @PathVariable Long predictionId,
            @RequestParam String rank,
            @RequestParam String candidateText,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            adminService.createCandidate(predictionId, rank, candidateText);
            redirectAttributes.addFlashAttribute("adminMessage", "Candidato creado correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo crear el candidato.");
        }
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    @PostMapping("/admin/trials/{trialId}/candidates/{candidateId}")
    public String updateCandidate(
            @PathVariable Long trialId,
            @PathVariable Long candidateId,
            @RequestParam String rank,
            @RequestParam String candidateText,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        try {
            adminService.updateCandidate(candidateId, rank, candidateText);
            redirectAttributes.addFlashAttribute("adminMessage", "Candidato actualizado correctamente.");
        } catch (RuntimeException exception) {
            redirectAttributes.addFlashAttribute("adminError", "No se pudo actualizar el candidato.");
        }
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    @PostMapping("/admin/trials/{trialId}/candidates/{candidateId}/delete")
    public String deleteCandidate(
            @PathVariable Long trialId,
            @PathVariable Long candidateId,
            HttpSession session,
            RedirectAttributes redirectAttributes
    ) {
        if (!isAdmin(session)) {
            return "redirect:/admin/login";
        }

        adminService.deleteCandidate(candidateId);
        redirectAttributes.addFlashAttribute("adminMessage", "Candidato eliminado correctamente.");
        return "redirect:/admin/trials/" + trialId + "/edit";
    }

    private String startAdminSession(AppUser admin, HttpSession session) {
        session.setAttribute(ADMIN_ID, admin.getId());
        session.setAttribute(ADMIN_NAME, admin.getDisplayName());
        return "redirect:/admin";
    }

    private boolean isAdmin(HttpSession session) {
        return session.getAttribute(ADMIN_ID) != null;
    }
}
